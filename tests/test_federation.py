"""端到端联邦训练测试：IID/non-IID、规模差异、掉线、失败、
聚合加权、checkpoint 恢复与确定性。

所有测试均在终端执行并打印 loss/accuracy。
"""
import numpy as np
import pytest

from fedavg_sim import (
    FedAvgSimulator,
    SimConfig,
    build_federated_dataset,
)
from fedavg_sim.aggregation import fed_avg
from fedavg_sim.client import ClientUpdate
from fedavg_sim.model import LocalTrainingError, Params, local_train


def make_env(
    *,
    seed=42,
    iid=False,
    n_clients=6,
    n_samples=2000,
    sizes_dirichlet=None,
    client_sizes=None,
    label_alpha=0.5,
):
    return build_federated_dataset(
        n_clients=n_clients,
        n_samples=n_samples,
        n_features=10,
        iid=iid,
        label_alpha=label_alpha,
        sizes_dirichlet=sizes_dirichlet,
        client_sizes=client_sizes,
        seed=seed,
    )


def run_and_report(name, history, global_params):
    """测试辅助：打印每轮的 loss/accuracy（满足终端可观测要求）。"""
    print(f"\n--- {name} ---")
    for m in history:
        tag = "SKIP" if m.skipped else "ok"
        tl = "  nan " if m.train_loss is None else f"{m.train_loss:.4f}"
        print(
            f"  round {m.round:02d} [{tag}] contrib={list(m.contributing_ids)} "
            f"n_eff={m.total_effective_samples} train_loss={tl} "
            f"test_loss={m.test_loss:.4f} acc={m.test_accuracy:.4f}"
        )
    last = [m for m in history if not m.skipped][-1]
    print(f"  FINAL test_loss={last.test_loss:.4f} test_accuracy={last.test_accuracy:.4f}")
    return last


# --------------------------------------------------------------------- #
# 学习有效性：IID 与 non-IID
# --------------------------------------------------------------------- #
def test_iid_training_improves_loss_and_accuracy():
    clients, X_test, y_test = make_env(iid=True, seed=1)
    cfg = SimConfig(seed=1, rounds=15, n_participants=4, local_epochs=3,
                    learning_rate=0.2, batch_size=32)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    first = next(m for m in history if not m.skipped)
    last = run_and_report("IID", history, sim.global_params)
    assert last.test_loss < first.test_loss - 0.05
    assert last.test_accuracy > first.test_accuracy
    assert last.test_accuracy > 0.75


def test_noniid_training_still_learns():
    clients, X_test, y_test = make_env(iid=False, label_alpha=0.3, seed=2)
    cfg = SimConfig(seed=2, rounds=20, n_participants=5, local_epochs=3,
                    learning_rate=0.2)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    first = next(m for m in history if not m.skipped)
    last = run_and_report("non-IID", history, sim.global_params)
    assert last.test_loss < first.test_loss
    assert last.test_accuracy > 0.7


# --------------------------------------------------------------------- #
# 不同规模 client：FedAvg 必须按有效样本数加权
# --------------------------------------------------------------------- #
def test_unequal_client_sizes_weighted_correctly():
    sizes = [2000, 300, 99, 1]  # 训练集总量 2400
    clients, X_test, y_test = make_env(
        n_clients=4, n_samples=3000, client_sizes=sizes, iid=True, seed=3
    )
    cfg = SimConfig(seed=3, rounds=5, n_participants=4, local_epochs=1)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    m = history[0]
    assert m.total_effective_samples == sum(sizes)
    # 手工复现第 1 轮聚合，验证严格按 n_k 加权
    manual_updates: list[ClientUpdate] = []
    _, _, failure_rng, train_ss = sim._round_rngs(1)
    train_seeds = train_ss.spawn(len(m.sampled_ids))
    for cid, ss in zip(m.sampled_ids, train_seeds):
        upd = sim._clients[cid].train(
            Params.zeros(10), round_idx=1, seed=int(ss.generate_state(1)[0])
        )
        manual_updates.append(upd)
    agg = fed_avg(manual_updates)
    assert agg.total_effective_samples == sum(sizes)
    # 最大 client（2000/2400 权重）应主导聚合结果
    w_big = next(u for u in manual_updates if u.effective_n_samples == 2000).params.w
    w_small = next(u for u in manual_updates if u.effective_n_samples == 1).params.w
    assert not np.allclose(agg.params.w, np.mean([u.params.w for u in manual_updates], axis=0))
    dist_big = np.abs(agg.params.w - w_big).sum()
    dist_small = np.abs(agg.params.w - w_small).sum()
    assert dist_big < dist_small

    run_and_report("unequal-sizes", history, sim.global_params)


# --------------------------------------------------------------------- #
# 掉线
# --------------------------------------------------------------------- #
def test_dropouts_do_not_break_round():
    clients, X_test, y_test = make_env(n_clients=6, seed=4)
    cfg = SimConfig(seed=4, rounds=12, n_participants=5, dropout_prob=0.4,
                    local_epochs=2, learning_rate=0.2)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    rounds_with_dropout = [m for m in history if m.dropped_ids]
    assert rounds_with_dropout, "0.4 掉线概率下 12 轮应至少有一轮出现掉线"
    assert all(not m.skipped for m in history)
    # 掉线 client 绝不出现在贡献者中
    for m in history:
        assert not set(m.dropped_ids) & set(m.contributing_ids)
    # 贡献者总样本数 = 各贡献 client 实际大小之和
    size_by_id = {c.client_id: c.n_samples for c in clients}
    for m in history:
        assert m.total_effective_samples == sum(
            size_by_id[cid] for cid in m.contributing_ids
        )
    run_and_report("dropout", history, sim.global_params)


def test_permanent_offline_clients_never_sampled():
    clients, X_test, y_test = make_env(n_clients=5, seed=5)
    cfg = SimConfig(seed=5, rounds=10, n_participants=4,
                    offline_client_ids=(0, 4))
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()
    for m in history:
        assert 0 not in m.sampled_ids and 4 not in m.sampled_ids


# --------------------------------------------------------------------- #
# 训练失败
# --------------------------------------------------------------------- #
def test_failing_clients_are_skipped_but_round_succeeds():
    clients, X_test, y_test = make_env(n_clients=6, seed=6)
    cfg = SimConfig(seed=6, rounds=12, n_participants=5, failure_prob=0.35,
                    always_fail_client_ids=(2,), local_epochs=2, learning_rate=0.2)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    rounds_with_fail = [m for m in history if m.failed_ids]
    assert rounds_with_fail, "应出现随机失败/强制失败"
    assert any(2 in m.failed_ids for m in history), "client 2 被采样时必须失败"
    assert all(not m.skipped for m in history)
    for m in history:
        assert not set(m.failed_ids) & set(m.contributing_ids)
    last = run_and_report("failures", history, sim.global_params)
    assert last.test_accuracy > 0.65  # 部分失败下整体训练仍然有效


def test_all_participants_fail_round_is_skipped_and_training_continues():
    clients, X_test, y_test = make_env(n_clients=4, seed=7)
    cfg = SimConfig(seed=7, rounds=6, n_participants=2,
                    always_fail_client_ids=(0, 1, 2, 3))
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()

    assert all(m.skipped for m in history)
    # 全局模型保持零初始化
    np.testing.assert_array_equal(sim.global_params.w, np.zeros(10))
    assert sim.global_params.b == 0.0
    print("\n--- all-fail: 全部轮次跳过，模型未被破坏 ---")


def test_min_successful_participants_threshold():
    clients, X_test, y_test = make_env(n_clients=5, seed=8)
    # 5 个 client 中 3 个必失败，每轮采样 4 个，要求至少 3 个成功 -> 每轮跳过
    cfg = SimConfig(seed=8, rounds=4, n_participants=4,
                    always_fail_client_ids=(0, 1, 2),
                    min_successful_participants=3)
    sim = FedAvgSimulator(clients, X_test, y_test, cfg)
    history = sim.run()
    assert all(m.skipped for m in history)


def test_single_failure_does_not_change_other_clients_contribution():
    """同一轮同样的参与者：client A 成功/失败两种情况下，
    其余 client 的本地更新参数必须逐位相同。"""
    clients, X_test, y_test = make_env(n_clients=5, seed=10)

    def train_once(always_fail=()):
        cfg = SimConfig(seed=10, rounds=3, n_participants=3,
                        always_fail_client_ids=tuple(always_fail))
        sim = FedAvgSimulator(clients, X_test, y_test, cfg)
        metrics = sim.run()[0]
        # 用该轮派生的训练种子，从零模型手工复现每个成功 client 的更新
        train_ss = sim._round_rngs(1)[3]
        seeds = train_ss.spawn(len(metrics.sampled_ids))
        updates = {}
        for cid, ss in zip(metrics.sampled_ids, seeds):
            if cid in metrics.failed_ids:
                continue
            updates[cid] = sim._clients[cid].train(
                Params.zeros(10), round_idx=1,
                seed=int(ss.generate_state(1)[0]),
            ).params
        return metrics, updates

    m_ok, params_ok = train_once()
    victim = m_ok.sampled_ids[0]
    m_fail, params_others = train_once(always_fail=[victim])

    assert m_ok.sampled_ids == m_fail.sampled_ids  # 采样确定，与失败配置无关
    assert victim in m_fail.failed_ids and victim not in m_fail.contributing_ids
    for cid in m_fail.contributing_ids:
        np.testing.assert_allclose(
            params_others[cid].w, params_ok[cid].w, rtol=0, atol=1e-12
        )
        assert params_others[cid].b == pytest.approx(params_ok[cid].b, abs=1e-12)


# --------------------------------------------------------------------- #
# Checkpoint：中断恢复后结果与连续运行一致
# --------------------------------------------------------------------- #
def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    clients, X_test, y_test = make_env(
        iid=False, n_clients=6, seed=11, sizes_dirichlet=0.5
    )
    cfg = SimConfig(seed=11, rounds=12, n_participants=4, local_epochs=2,
                    learning_rate=0.2, dropout_prob=0.15, failure_prob=0.1,
                    checkpoint_every=1)

    # 不中断连续运行
    full = FedAvgSimulator(clients, X_test, y_test, cfg)
    full_history = full.run()

    # 在第 5 轮后中断，再从 checkpoint 恢复
    ckpt_dir = tmp_path / "ckpt"
    interrupted = FedAvgSimulator(
        clients, X_test, y_test, cfg, checkpoint_dir=str(ckpt_dir)
    )
    interrupted.run(stop_after_round=5)
    assert interrupted.next_round == 6

    resumed = FedAvgSimulator.resume(str(ckpt_dir), clients, X_test, y_test)
    resumed_history = resumed.run()

    # 最终参数与指标逐元素一致（float64 + 确定性 RNG 派生，容差可给得很严）
    np.testing.assert_allclose(
        resumed.global_params.w, full.global_params.w, rtol=0, atol=1e-12
    )
    assert resumed.global_params.b == pytest.approx(full.global_params.b, abs=1e-12)
    assert resumed.next_round == 13 and len(resumed.history) == 12
    for a, b in zip(resumed.history, full_history):
        assert a.sampled_ids == b.sampled_ids
        assert a.contributing_ids == b.contributing_ids
        assert a.skipped == b.skipped
        assert (a.train_loss is None) == (b.train_loss is None)
        if a.train_loss is not None:
            assert a.train_loss == pytest.approx(b.train_loss, abs=1e-12)
            assert a.test_loss == pytest.approx(b.test_loss, abs=1e-12)
            assert a.test_accuracy == b.test_accuracy
    run_and_report("resume(final)", resumed.history, resumed.global_params)


def test_checkpoint_resume_at_every_round(tmp_path):
    """在每一个可能的轮次边界恢复，最终模型都必须与连续运行一致。"""
    clients, X_test, y_test = make_env(iid=True, n_clients=5, n_samples=1500, seed=12)
    cfg = SimConfig(seed=12, rounds=8, n_participants=4, learning_rate=0.15,
                    checkpoint_every=1)
    full = FedAvgSimulator(clients, X_test, y_test, cfg)
    full.run()

    for stop in range(1, 8):
        ckpt = tmp_path / f"ckpt_{stop}"
        sim = FedAvgSimulator(
            clients, X_test, y_test, cfg, checkpoint_dir=str(ckpt),
        )
        sim.run(stop_after_round=stop)
        resumed = FedAvgSimulator.resume(str(ckpt), clients, X_test, y_test)
        resumed.run()
        np.testing.assert_allclose(
            resumed.global_params.w, full.global_params.w, rtol=0, atol=1e-12
        )
        assert len(resumed.history) == 8
    print("\n--- checkpoint: 1..7 任意轮次恢复均与连续运行逐位一致 ---")


def test_checkpoint_files_are_local_only(tmp_path):
    clients, X_test, y_test = make_env(n_clients=4, seed=13)
    ckpt = tmp_path / "ck"
    cfg = SimConfig(seed=13, rounds=3, n_participants=3, checkpoint_every=1)
    FedAvgSimulator(clients, X_test, y_test, cfg, checkpoint_dir=str(ckpt)).run()
    names = sorted(p.name for p in ckpt.iterdir())
    assert names == ["config.json", "model.npz", "state.json"]


# --------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------- #
def test_determinism_same_seed_identical_runs():
    histories, finals = [], []
    for _ in range(2):
        clients, X_test, y_test = make_env(
            iid=False, n_clients=7, seed=21, sizes_dirichlet=0.4
        )
        cfg = SimConfig(seed=21, rounds=10, n_participants=4, dropout_prob=0.2,
                        failure_prob=0.15, local_epochs=2, learning_rate=0.2)
        sim = FedAvgSimulator(clients, X_test, y_test, cfg)
        h = sim.run()
        histories.append(h)
        finals.append(sim.global_params)

    h1, h2 = histories
    for a, b in zip(h1, h2):
        assert a.sampled_ids == b.sampled_ids
        assert a.dropped_ids == b.dropped_ids
        assert a.failed_ids == b.failed_ids
        assert a.contributing_ids == b.contributing_ids
        if a.train_loss is not None:
            assert a.train_loss == b.train_loss
            assert a.test_loss == b.test_loss
    np.testing.assert_array_equal(finals[0].w, finals[1].w)
    assert finals[0].b == finals[1].b
    run_and_report("determinism", h1, finals[0])


def test_different_seeds_change_sampling():
    def sampled_for(seed):
        clients, X_test, y_test = make_env(n_clients=8, seed=30 + seed)
        cfg = SimConfig(seed=seed, rounds=5, n_participants=3)
        return [m.sampled_ids for m in FedAvgSimulator(clients, X_test, y_test, cfg).run()]

    a = sampled_for(100)
    b = sampled_for(200)
    assert a != b  # 极小概率相同，但 5 轮 × 3/8 下实际不会


# --------------------------------------------------------------------- #
# 本地训练失败的底层路径
# --------------------------------------------------------------------- #
def test_local_train_rejects_empty_and_bad_labels():
    with pytest.raises(LocalTrainingError):
        local_train(Params.zeros(3), np.zeros((0, 3)), np.zeros(0))
    with pytest.raises(LocalTrainingError):
        local_train(Params.zeros(2), np.ones((4, 2)), np.array([0, 1, 2, 0]))

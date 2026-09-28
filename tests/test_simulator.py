"""端到端 FedAvg、失败隔离、确定性与 checkpoint 恢复测试。"""
import os

import numpy as np
import pytest

from fedavg.simulator import SimConfig, Simulator


def _iid_config(**overrides) -> SimConfig:
    cfg = SimConfig(
        seed=42, rounds=20, n_clients=5, n_samples=2000, n_features=6,
        iid=True, participation_frac=0.8, local_epochs=2, lr=0.1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _non_iid_config(**overrides) -> SimConfig:
    cfg = _iid_config(iid=False, alpha=0.3)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_iid_end_to_end_converges(capsys):
    sim = Simulator(_iid_config())
    history = sim.run(verbose=True)
    cap = capsys.readouterr()
    assert "loss=" in cap.out and "acc=" in cap.out  # 终端输出 loss/accuracy

    losses = [r.loss for r in history]
    accs = [r.accuracy for r in history]
    assert losses[-1] < losses[0]
    assert accs[-1] > 0.80
    # 损失总体单调下降（允许轻微波动，比较首尾四分位）。
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    print(f"\nIID 最终: loss={losses[-1]:.4f} accuracy={accs[-1]:.4f}")


def test_non_iid_end_to_end_learns(capsys):
    sim = Simulator(_non_iid_config(rounds=30))
    history = sim.run(verbose=True)
    cap = capsys.readouterr()
    assert "loss=" in cap.out
    final = history[-1]
    assert final.accuracy > 0.70
    assert final.loss < history[0].loss
    print(f"\nnon-IID(alpha=0.3) 最终: loss={final.loss:.4f} "
          f"accuracy={final.accuracy:.4f}")


def test_heterogeneous_client_sizes_weighted_aggregation():
    # 规模差异巨大的客户端：聚合必须按有效样本数加权（由 n_effective_total 体现）。
    sim = Simulator(
        _iid_config(n_clients=6, heterogeneous_sizes=True, rounds=10, seed=21)
    )
    history = sim.run(verbose=False)
    sizes = {cid: c.n_samples for cid, c in sim.clients.items()}
    assert max(sizes.values()) > 3 * min(sizes.values())
    for r in history:
        # 每轮 n_eff 必须恰好等于成功客户端的真实样本数之和。
        expected = sum(sizes[cid] for cid in r.succeeded)
        assert r.n_effective_total == expected
    print("\n异构规模每轮 n_effective:",
          [r.n_effective_total for r in history])


def test_random_drops_and_failures_do_not_break_round(capsys):
    # 每客户端 25% 掉线 + 20% 失败概率；某轮全员失败时允许跳过，
    # 但训练流程与最终收敛不得被破坏。
    cfg = _iid_config(rounds=25, drop_prob=0.25, fail_prob=0.2, seed=55)
    sim = Simulator(cfg)
    history = sim.run(verbose=True)
    cap = capsys.readouterr()
    n_drop = sum(len(r.dropped) for r in history)
    n_fail = sum(len(r.failed) for r in history)
    assert n_drop + n_fail > 0  # 概率足够高，必然出现失败
    # 没有失败客户端污染聚合：所有轮次参数有限，分类集合互斥。
    for r in history:
        assert np.isfinite(r.loss)
        assert set(r.dropped).isdisjoint(r.succeeded)
        assert set(r.failed).isdisjoint(r.succeeded)
        assert set(r.dropped).isdisjoint(r.failed)
    # 25 轮中绝大多数轮次必须成功聚合（全员失败概率很低）。
    completed = [r for r in history if not r.skipped]
    assert len(completed) >= 22
    # 即使有失败与跳过，FedAvg 最终仍应学习到有效模型。
    assert history[-1].accuracy > 0.75
    print(f"\n随机失败: 累计掉线 {n_drop} 次, 训练失败 {n_fail} 次, "
          f"跳过 {len(history) - len(completed)} 轮, "
          f"最终 acc={history[-1].accuracy:.4f}")
    assert "drop=" in cap.out  # 终端逐轮输出了失败信息


def test_forced_partial_failures_aggregation_uses_survivors():
    # 第 2 轮全员参与，3 个被强制失败，剩余 2 个必须正常完成聚合。
    schedule = {2: {0: "fail", 1: "drop", 4: "fail"}}
    cfg = _iid_config(rounds=5, seed=5, participation_frac=1.0)
    sim = Simulator(cfg, failure_schedule=schedule)
    history = sim.run(verbose=False)
    rec = history[2]
    assert sorted(rec.selected) == [0, 1, 2, 3, 4]
    assert sorted(rec.failed) == [0, 4]
    assert rec.dropped == [1]
    assert sorted(rec.succeeded) == [2, 3]
    assert not rec.skipped
    # 聚合权重仅来自存活客户端。
    sizes = {cid: c.n_samples for cid, c in sim.clients.items()}
    assert rec.n_effective_total == sizes[2] + sizes[3]


def test_all_clients_fail_round_is_skipped():
    # 全员参与的某一轮全部失败：该轮跳过、参数不变，训练继续。
    cfg = _iid_config(rounds=6, participation_frac=1.0, seed=9)
    schedule = {3: {0: "drop", 1: "fail", 2: "drop", 3: "fail", 4: "drop"}}
    sim = Simulator(cfg, failure_schedule=schedule)
    history = sim.run(verbose=False)
    rec3 = history[3]
    assert rec3.skipped
    assert rec3.succeeded == []
    assert rec3.n_effective_total == 0
    # 跳过轮的参数与上一轮完全相同（指标也相同）。
    assert rec3.loss == history[2].loss
    assert rec3.accuracy == history[2].accuracy
    # 下一轮恢复正常。
    assert not history[4].skipped
    assert len(history[4].succeeded) >= 1
    print("\n全员失败轮被跳过，前后 loss:",
          history[2].loss, rec3.loss, history[4].loss)


def test_determinism_same_seed_identical_runs():
    def execute(seed):
        sim = Simulator(_iid_config(seed=seed, rounds=12, drop_prob=0.2,
                                    fail_prob=0.1))
        sim.run(verbose=False)
        return sim

    a = execute(77)
    b = execute(77)
    for ra, rb in zip(a.history, b.history):
        assert ra.selected == rb.selected
        assert ra.succeeded == rb.succeeded
        assert ra.dropped == rb.dropped
        assert ra.failed == rb.failed
        assert ra.loss == rb.loss
        assert ra.accuracy == rb.accuracy
    np.testing.assert_array_equal(
        a.global_params["w"], b.global_params["w"]
    )

    c = execute(78)
    # 不同种子应产生不同的参与者序列。
    assert any(
        x.selected != y.selected
        for x, y in zip(a.history, c.history)
    )
    print("\n同种子两次运行最终权重完全一致；"
          f"A acc={a.history[-1].accuracy:.4f}, "
          f"C(seed=78) acc={c.history[-1].accuracy:.4f}")


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    ckpt = str(tmp_path / "ckpt.pkl")

    # 参考：不中断执行 20 轮。
    ref_sim = Simulator(_iid_config(rounds=20, seed=33))
    ref_sim.run(verbose=False)

    # 分段执行：先跑 7 轮并存 checkpoint，再恢复跑完 20 轮。
    part = Simulator(_iid_config(rounds=7, seed=33, checkpoint_path=ckpt))
    part.run(verbose=False)
    assert os.path.exists(ckpt)

    resumed = Simulator.resume(
        ckpt, config_override=_iid_config(rounds=20, seed=33, checkpoint_path=ckpt)
    )
    assert resumed.round == 7
    resumed_history = resumed.run(verbose=False)
    assert resumed.round == 20
    assert len(resumed.history) == 20

    # 恢复后的每一轮（含采样、成功/失败、指标）必须与不中断执行逐轮一致。
    for r_ref, r_res in zip(ref_sim.history, resumed.history):
        assert r_ref.selected == r_res.selected
        assert r_ref.succeeded == r_res.succeeded
        assert r_ref.loss == r_res.loss
        assert r_ref.accuracy == r_res.accuracy
    np.testing.assert_allclose(
        resumed.global_params["w"], ref_sim.global_params["w"], rtol=1e-12
    )
    np.testing.assert_allclose(
        resumed.global_params["b"], ref_sim.global_params["b"], rtol=1e-12
    )
    loss, acc = resumed.evaluate()
    print(f"\nCheckpoint 恢复最终: loss={loss:.4f} accuracy={acc:.4f}，"
          "与不中断执行逐轮一致")


def test_checkpoint_resume_at_multiple_points(tmp_path):
    for stop_round in (1, 5, 10, 19):
        ckpt = str(tmp_path / f"ckpt_{stop_round}.pkl")
        ref = Simulator(_iid_config(rounds=20, seed=123))
        ref.run(verbose=False)

        part = Simulator(
            _iid_config(rounds=stop_round, seed=123, checkpoint_path=ckpt)
        )
        part.run(verbose=False)
        resumed = Simulator.resume(
            ckpt,
            config_override=_iid_config(
                rounds=20, seed=123, checkpoint_path=ckpt
            ),
        )
        resumed.run(verbose=False)
        np.testing.assert_allclose(
            resumed.global_params["w"], ref.global_params["w"], rtol=1e-12
        )


def test_checkpoint_resume_with_failures(tmp_path):
    # 掉线/失败场景下恢复，结果同样必须一致。
    ckpt = str(tmp_path / "ckpt_fail.pkl")
    cfg_kwargs = dict(rounds=16, seed=88, drop_prob=0.2, fail_prob=0.2)

    ref = Simulator(_iid_config(**cfg_kwargs))
    ref.run(verbose=False)

    part = Simulator(
        _iid_config(**{**cfg_kwargs, "rounds": 6, "checkpoint_path": ckpt})
    )
    part.run(verbose=False)
    resumed = Simulator.resume(
        ckpt, config_override=_iid_config(
            **{**cfg_kwargs, "checkpoint_path": ckpt}
        )
    )
    resumed.run(verbose=False)

    for r_ref, r_res in zip(ref.history, resumed.history):
        assert r_ref.dropped == r_res.dropped
        assert r_ref.failed == r_res.failed
        assert r_ref.loss == r_res.loss
    np.testing.assert_allclose(
        resumed.global_params["w"], ref.global_params["w"], rtol=1e-12
    )


def test_checkpoint_rejects_config_mismatch(tmp_path):
    ckpt = str(tmp_path / "ckpt.pkl")
    part = Simulator(_iid_config(rounds=3, seed=1, checkpoint_path=ckpt))
    part.run(verbose=False)
    with pytest.raises(ValueError):
        Simulator.resume(ckpt, config_override=_iid_config(rounds=10, seed=2))

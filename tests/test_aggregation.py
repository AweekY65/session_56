"""FedAvg 聚合计算测试：有效样本数加权、拒绝非法更新、空更新语义。"""
import numpy as np
import pytest

from fedavg_sim import ClientUpdate, NoValidUpdatesError, Params, fed_avg


def _update(client_id: int, w: np.ndarray, b: float, n: int, loss: float = 0.5) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id, params=Params(w=np.asarray(w, dtype=np.float64), b=b),
        effective_n_samples=n, train_loss=loss,
    )


def test_weighted_by_effective_samples_not_simple_mean():
    """两个规模悬殊的 client：结果必须按 n 加权，而不是简单平均。"""
    small = _update(0, w=np.array([1.0, 2.0]), b=0.0, n=10)
    large = _update(1, w=np.array([5.0, 6.0]), b=2.0, n=90)

    agg = fed_avg([small, large])

    expected_w = 0.1 * np.array([1.0, 2.0]) + 0.9 * np.array([5.0, 6.0])
    np.testing.assert_allclose(agg.params.w, expected_w, rtol=0, atol=1e-12)
    assert agg.params.b == pytest.approx(1.8, abs=1e-12)
    assert agg.total_effective_samples == 100
    assert agg.contributing_client_ids == (0, 1)


def test_matches_manual_formula_with_three_clients():
    updates = [
        _update(3, w=np.array([2.0]), b=1.0, n=100),
        _update(1, w=np.array([4.0]), b=3.0, n=300),
        _update(2, w=np.array([6.0]), b=5.0, n=600),
    ]
    agg = fed_avg(updates)
    # 总权重 1000
    assert agg.params.w[0] == pytest.approx((2 * 100 + 4 * 300 + 6 * 600) / 1000)
    assert agg.params.b == pytest.approx((1 * 100 + 3 * 300 + 5 * 600) / 1000)
    # 结果与传入顺序无关（按 client_id 排序）
    agg_rev = fed_avg(list(reversed(updates)))
    np.testing.assert_array_equal(agg_rev.params.w, agg.params.w)
    assert agg_rev.params.b == agg.params.b


def test_rejects_non_finite_update_but_keeps_valid_ones():
    bad = _update(0, w=np.array([np.nan, 1.0]), b=0.0, n=50)
    inf = _update(2, w=np.array([1.0, np.inf]), b=0.0, n=50)
    good = _update(1, w=np.array([2.0, 2.0]), b=1.0, n=200)

    agg = fed_avg([bad, good, inf])
    assert agg.contributing_client_ids == (1,)
    assert set(agg.rejected_client_ids) == {0, 2}
    np.testing.assert_allclose(agg.params.w, np.array([2.0, 2.0]))
    assert agg.params.b == 1.0


def test_rejects_nonpositive_sample_counts():
    # n<=0 在 ClientUpdate 构造时即被拒绝；手工绕过构造校验模拟非法来源。
    bad_update = ClientUpdate.__new__(ClientUpdate)
    bad_update.client_id = 0
    bad_update.params = Params(np.array([9.0]), 9.0)
    bad_update.effective_n_samples = 0
    bad_update.train_loss = 0.1

    good = _update(1, w=np.array([1.0]), b=1.0, n=10)
    agg = fed_avg([bad_update, good])
    assert agg.contributing_client_ids == (1,)
    assert agg.rejected_client_ids == (0,)


def test_all_invalid_raises():
    with pytest.raises(NoValidUpdatesError):
        fed_avg([])

    bad = _update(0, w=np.array([np.nan]), b=0.0, n=10)
    with pytest.raises(NoValidUpdatesError):
        fed_avg([bad])


def test_weights_sum_to_one_and_total_samples():
    sizes = [1, 2, 3, 4, 90]
    base = np.zeros(2)
    updates = [
        _update(i, w=base + i, b=float(i), n=n)
        for i, n in enumerate(sizes)
    ]
    agg = fed_avg(updates)
    assert agg.total_effective_samples == sum(sizes)
    # 等价于权重之和为 1：加权结果落在各 client 参数的凸包内
    assert 0.0 <= agg.params.w[0] <= 4.0
    assert agg.params.b == pytest.approx(
        sum(i * n for i, n in enumerate(sizes)) / sum(sizes)
    )

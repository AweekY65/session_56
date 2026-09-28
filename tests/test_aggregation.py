"""FedAvg 加权聚合与本地训练逻辑测试。"""
import numpy as np
import pytest

from fedavg.client import (
    Client,
    ClientDroppedError,
    ClientFailureError,
    ClientUpdate,
    federated_average,
    valid_update,
)
from fedavg.data import ClientData, make_classification
from fedavg.model import LogisticRegression


def _update(cid: int, w: np.ndarray, n: int, b: float = 0.0) -> ClientUpdate:
    return ClientUpdate(
        client_id=cid,
        params={"w": w.astype(float), "b": np.array(float(b))},
        n_effective=n,
    )


def test_fedavg_weighted_by_sample_counts():
    # 两个客户端：100 和 300 个样本，参数 0 和 10。
    # 正确聚合 = 0.25*0 + 0.75*10 = 7.5（不是简单平均 5）。
    u1 = _update(0, np.zeros(3), 100, b=0.0)
    u2 = _update(1, np.full(3, 10.0), 300, b=10.0)
    agg = federated_average([u1, u2])
    np.testing.assert_allclose(agg["w"], np.full(3, 7.5))
    np.testing.assert_allclose(agg["b"], np.array(7.5))


def test_fedavg_many_clients_weights_normalize():
    rng = np.random.default_rng(0)
    sizes = [10, 50, 130, 7, 42]
    updates = [
        _update(i, rng.normal(size=4) + i, n) for i, n in enumerate(sizes)
    ]
    agg = federated_average(updates)
    total = sum(sizes)
    expected = sum((n / total) * u.params["w"] for n, u in zip(sizes, updates))
    np.testing.assert_allclose(agg["w"], expected, rtol=1e-12)


def test_fedavg_equal_sizes_equals_manual_average():
    u1 = _update(0, np.array([1.0, 2.0]), 50)
    u2 = _update(1, np.array([3.0, 4.0]), 50)
    agg = federated_average([u1, u2])
    np.testing.assert_allclose(agg["w"], np.array([2.0, 3.0]))


def test_fedavg_empty_and_nonpositive_weights():
    assert federated_average([]) is None
    bad = _update(0, np.ones(2), 0)
    assert federated_average([bad]) is None


def test_valid_update_rejects_nonfinite():
    good = _update(0, np.ones(2), 10)
    assert valid_update(good)
    bad_nan = _update(1, np.array([np.nan, 1.0]), 10)
    bad_inf = _update(2, np.array([np.inf, 1.0]), 10)
    bad_n = _update(3, np.ones(2), -1)
    assert not valid_update(bad_nan)
    assert not valid_update(bad_inf)
    assert not valid_update(bad_n)


def _make_client(cid, n=200, drop_prob=0.0, fail_prob=0.0, seed=0):
    data, _ = make_classification(1000, 5, seed=100)
    cdata = ClientData(cid, data.x[:n], data.y[:n])
    model = LogisticRegression(5)
    return Client(
        cid, cdata, model, local_epochs=3, lr=0.1,
        drop_prob=drop_prob, fail_prob=fail_prob,
        rng=np.random.default_rng(seed),
    )


def test_local_training_decreases_loss():
    client = _make_client(0)
    params = client.model.init_params(np.random.default_rng(0))
    before = client.local_loss(params)
    update = client.train(params)
    after = client.model.loss(update.params, client.data.x, client.data.y)
    assert after < before
    assert update.n_effective == client.n_samples
    print(f"\n本地训练: loss {before:.4f} -> {after:.4f}, "
          f"n_effective={update.n_effective}")


def test_client_drop_raises_dropped_error():
    # drop_prob=1 => 必然掉线。
    client = _make_client(0, drop_prob=1.0, seed=1)
    params = client.model.init_params(np.random.default_rng(0))
    with pytest.raises(ClientDroppedError):
        client.train(params)


def test_client_failure_raises_failure_error():
    client = _make_client(0, fail_prob=1.0, seed=1)
    params = client.model.init_params(np.random.default_rng(0))
    with pytest.raises(ClientFailureError):
        client.train(params)


def test_zero_prob_never_drops_or_fails():
    client = _make_client(0, drop_prob=0.0, fail_prob=0.0, seed=2)
    params = client.model.init_params(np.random.default_rng(0))
    # 任何随机数下都不应失败。
    for _ in range(20):
        client.rng = np.random.default_rng()
        update = client.train(params)
        assert np.all(np.isfinite(update.params["w"]))

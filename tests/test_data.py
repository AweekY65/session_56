"""数据生成与 IID/non-IID 划分测试。"""
import numpy as np

from fedavg.data import (
    iid_split,
    make_classification,
    make_clients,
    non_iid_split,
)


def test_make_classification_shapes_and_labels():
    data, w_true = make_classification(n_samples=500, n_features=4, seed=1)
    assert data.x.shape == (500, 4)
    assert data.y.shape == (500,)
    assert set(np.unique(data.y)).issubset({0, 1})
    assert w_true.shape == (4,)
    # 两类都应出现（w_scale=1.5，标签不会退化）。
    assert set(np.unique(data.y)) == {0, 1}


def test_data_deterministic_under_seed():
    d1, w1 = make_classification(1000, 5, seed=123)
    d2, w2 = make_classification(1000, 5, seed=123)
    np.testing.assert_array_equal(d1.x, d2.x)
    np.testing.assert_array_equal(d1.y, d2.y)
    np.testing.assert_array_equal(w1, w2)

    d3, _ = make_classification(1000, 5, seed=124)
    assert not np.array_equal(d1.x, d3.x)


def test_train_test_share_same_task():
    # 测试集必须来自同一隐藏模型，否则评估无意义。
    _, train, test, w_true = make_clients(n_clients=3, seed=99)
    from fedavg.model import LogisticRegression

    model = LogisticRegression(train.x.shape[1])
    oracle = {"w": w_true, "b": np.zeros(())}
    assert model.accuracy(oracle, test.x, test.y) > 0.85


def test_iid_split_partitions_all_samples():
    data, _ = make_classification(1000, 4, seed=3)
    clients = iid_split(data, n_clients=4, seed=0)
    assert len(clients) == 4
    assert sum(c.n_samples for c in clients.values()) == 1000
    # 无重叠：特征为连续正态分布，重复概率为 0 —— 拼接后应仍为 1000 个唯一行。
    all_rows = np.concatenate([c.x for c in clients.values()])
    unique_rows = np.unique(all_rows, axis=0)
    assert unique_rows.shape[0] == 1000


def test_iid_label_distributions_are_similar():
    data, _ = make_classification(4000, 6, seed=5)
    clients = iid_split(data, n_clients=4, seed=0)
    rates = [c.y.mean() for c in clients.values()]
    # IID：各客户端正类比例应接近全局比例。
    for r in rates:
        assert abs(r - data.y.mean()) < 0.07


def test_non_iid_split_is_more_imbalanced_than_iid():
    data, _ = make_classification(4000, 6, seed=5)
    iid_clients = iid_split(data, n_clients=5, seed=0)
    nid_clients = non_iid_split(data, n_clients=5, alpha=0.1, seed=0)

    iid_spread = max(c.y.mean() for c in iid_clients.values()) - min(
        c.y.mean() for c in iid_clients.values()
    )
    nid_rates = [c.y.mean() for c in nid_clients.values()]
    nid_spread = max(nid_rates) - min(nid_rates)
    assert nid_spread > iid_spread
    # alpha 很小时，至少有一个客户端被单一类别主导。
    assert any(r < 0.15 or r > 0.85 for r in nid_rates)


def test_non_iid_small_alpha_extreme_skew():
    data, _ = make_classification(2000, 6, seed=8)
    clients = non_iid_split(data, n_clients=4, alpha=0.05, seed=0)
    # 所有样本仍被完整划分。
    assert sum(c.n_samples for c in clients.values()) == 2000
    # 没有空客户端（极端 alpha 下会自动补样本）。
    assert all(c.n_samples >= 1 for c in clients.values())


def test_heterogeneous_client_sizes():
    clients, _, _, _ = make_clients(
        n_clients=6, n_samples=2000, heterogeneous_sizes=True, seed=11
    )
    sizes = [c.n_samples for c in clients.values()]
    assert sum(sizes) == 2000
    # 对数正态规模应产生明显异构（最大至少是最小的 3 倍）。
    assert max(sizes) > 3 * min(sizes)


def test_split_determinism():
    data, _ = make_classification(1000, 4, seed=2)
    a = iid_split(data, 3, seed=7)
    b = iid_split(data, 3, seed=7)
    for cid in a:
        np.testing.assert_array_equal(a[cid].x, b[cid].x)

    c = non_iid_split(data, 3, alpha=0.3, seed=7)
    d = non_iid_split(data, 3, alpha=0.3, seed=7)
    for cid in c:
        np.testing.assert_array_equal(c[cid].x, d[cid].x)

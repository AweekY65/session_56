"""数据生成与 IID / non-IID 划分测试。"""
import numpy as np

from fedavg_sim import build_federated_dataset
from fedavg_sim.data import generate_classification_data


def test_all_samples_partitioned_without_overlap():
    clients, X_test, y_test = build_federated_dataset(
        n_clients=5, n_samples=1000, seed=7
    )
    n_train = sum(c.n_samples for c in clients)
    assert n_train + y_test.shape[0] == 1000
    assert all(c.X.shape[0] == c.y.shape[0] for c in clients)
    # 所有 client 数据都来自同一合成集，特征列一致
    assert all(c.X.shape[1] == 10 for c in clients)


def test_iid_partition_label_ratios_close_to_global():
    """IID 划分下各 client 的正类比例应接近总体比例。"""
    clients, _, _ = build_federated_dataset(
        n_clients=5, n_samples=4000, iid=True, seed=123
    )
    global_pos = np.concatenate([c.y for c in clients]).mean()
    ratios = [c.y.mean() for c in clients]
    assert all(abs(r - global_pos) < 0.12 for r in ratios), ratios


def test_iid_equal_sizes_by_default():
    clients, _, _ = build_federated_dataset(
        n_clients=4, n_samples=1000, iid=True, seed=1
    )
    sizes = [c.n_samples for c in clients]
    # 逐类别最大余数取整，client 间最多相差 2
    assert max(sizes) - min(sizes) <= 2
    assert sum(sizes) == 800  # 1000 - 200 测试


def test_non_iid_label_skew_is_more_imbalanced_than_iid():
    """alpha 很小时，non-IID 各 client 标签比例的离散度应明显大于 IID。"""
    niid, _, _ = build_federated_dataset(
        n_clients=6, n_samples=3000, iid=False, label_alpha=0.05, seed=99
    )
    iid, _, _ = build_federated_dataset(
        n_clients=6, n_samples=3000, iid=True, seed=99
    )
    spread = lambda cs: float(np.std([c.y.mean() for c in cs]))  # noqa: E731
    assert spread(niid) > spread(iid)
    # 极端倾斜下至少有一个 client 接近单标签
    assert min(min(c.y.mean(), 1 - c.y.mean()) for c in niid) < 0.15


def test_explicit_client_sizes_supported():
    # 训练集 = 3000 * 0.8 = 2400
    sizes = [2000, 300, 99, 1]
    clients, _, y_test = build_federated_dataset(
        n_clients=4, n_samples=3000, client_sizes=sizes, iid=True, seed=5
    )
    assert sorted(c.n_samples for c in clients) == [1, 99, 300, 2000]
    assert y_test.shape[0] == 600


def test_size_skew_dirichlet_produces_varied_sizes():
    clients, _, _ = build_federated_dataset(
        n_clients=8, n_samples=4000, sizes_dirichlet=0.1, seed=11
    )
    sizes = np.array([c.n_samples for c in clients], dtype=float)
    assert sizes.sum() == 3200
    assert sizes.max() / max(sizes.min(), 1.0) > 5.0  # 规模悬殊


def test_data_generation_is_deterministic():
    X1, y1 = generate_classification_data(n_samples=500, n_features=8, seed=2024)
    X2, y2 = generate_classification_data(n_samples=500, n_features=8, seed=2024)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

    X3, y3 = generate_classification_data(n_samples=500, n_features=8, seed=2025)
    assert not np.array_equal(X1, X3)


def test_partition_deterministic_and_labels_binary():
    a, xt_a, yt_a = build_federated_dataset(n_clients=5, label_alpha=0.3, seed=8)
    b, xt_b, yt_b = build_federated_dataset(n_clients=5, label_alpha=0.3, seed=8)
    for ca, cb in zip(a, b):
        np.testing.assert_array_equal(ca.X, cb.X)
        np.testing.assert_array_equal(ca.y, cb.y)
    np.testing.assert_array_equal(xt_a, xt_b)
    for c in a:
        assert set(np.unique(c.y)).issubset({0.0, 1.0})

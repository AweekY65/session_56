"""合成数据生成与 IID / non-IID 客户端划分。

所有数据都由代码用 numpy 生成，不下载任何数据集或模型。

合成数据：线性可分的二分类任务
    X ~ N(0, I) （每个样本 d 维特征）
    logit = X @ w_true + b_true + 噪声
    y = Bernoulli(sigmoid(logit)) ∈ {0, 1}

划分方式：
- IID: 每个 client 得到同样标签比例、规模相同（或按显式给定大小）的样本。
- non-IID 标签倾斜: 对每个 client 做一次 Dirichlet(alpha) 抽样得到该 client
  的标签比例，再按比例分配样本（alpha 越小，标签分布越倾斜）。
- 数量倾斜: 通过 sizes_dirichlet 为每个 client 抽取相对数据量，
  使不同 client 的规模差异很大（FedAvg 必须按有效样本数加权）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientData:
    """一个本地 client 持有的数据（全部在内存中，绝不离开本机）。"""

    client_id: int
    X: np.ndarray  # shape: (n_samples, n_features)
    y: np.ndarray  # shape: (n_samples,)，取值 {0, 1}

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """把 total 个物品按浮点权重确定性地分成整数份（最大余数法）。

    份额之和严格等于 total；并列时按索引顺序打破平局，保证可复现。
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    raw = weights / weights.sum() * total
    floors = np.floor(raw).astype(np.int64)
    remainder = total - int(floors.sum())
    frac = raw - floors
    # 余数从大到小分配；np.argsort 对并列返回索引升序，结果确定。
    order = np.argsort(-frac, kind="stable")
    for idx in order[:remainder]:
        floors[idx] += 1
    return floors


def generate_classification_data(
    n_samples: int = 2000,
    n_features: int = 10,
    seed: int = 42,
    noise_std: float = 0.5,
    w_true_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """生成合成二分类数据，返回 (X, y)。

    固定 seed 时生成结果确定性可复现。
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w_true = rng.standard_normal(n_features) * w_true_scale
    b_true = 0.3
    logits = X @ w_true + b_true + rng.standard_normal(n_samples) * noise_std
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probs).astype(np.float64)
    return X, y


def _split_indices_by_counts(
    indices: np.ndarray, counts: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
    """把一组索引按 counts 随机切成多份（不放回），返回每 client 的索引。"""
    perm = rng.permutation(indices)
    splits: list[np.ndarray] = []
    start = 0
    for count in counts:
        end = start + int(count)
        splits.append(perm[start:end])
        start = end
    return splits


def _iid_partition(
    y: np.ndarray,
    n_clients: int,
    client_sizes: np.ndarray | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n = y.shape[0]
    if client_sizes is None:
        sizes = _largest_remainder(np.ones(n_clients), n)
    else:
        sizes = np.asarray(client_sizes, dtype=np.int64)
        if sizes.sum() != n:
            raise ValueError(
                f"client_sizes 之和 ({sizes.sum()}) 必须等于样本数 ({n})"
            )

    # 把每个类别的样本独立洗牌后，按各 client 的容量做“交错发牌”，
    # 使每个 client 的类别占比严格等于其容量占比（IID）。
    buckets: list[list[int]] = [[] for _ in range(n_clients)]
    class_groups = [rng.permutation(np.where(y == label)[0]) for label in (0, 1)]
    rng.shuffle(class_groups)  # 不影响类别比例，仅决定先放哪一类
    for group in class_groups:
        # 每个 client 应从该类别拿到的数量 ≈ |class| * size_i / n
        per_client = _largest_remainder(sizes.astype(np.float64), int(group.shape[0]))
        start = 0
        for i in range(n_clients):
            end = start + int(per_client[i])
            buckets[i].extend(group[start:end].tolist())
            start = end
    return [np.sort(np.asarray(b, dtype=np.int64)) for b in buckets]


def _non_iid_partition(
    y: np.ndarray,
    n_clients: int,
    label_alpha: float,
    client_sizes: np.ndarray | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """non-IID 划分：每个 client 有独立的 Dirichlet 标签偏好。

    在固定 client 容量（client_sizes）的约束下，按
    p_{i,c} ~ Dirichlet(alpha) 给出的偏好把每个类别的样本分配给 client：
    每个类别在 client 间按“剩余容量 × 偏好”做最大余数法分配，
    因此样本全部分完、不重不漏，且 alpha 越小标签倾斜越严重。
    """
    n = y.shape[0]
    if client_sizes is None:
        sizes = _largest_remainder(np.ones(n_clients), n)
    else:
        sizes = np.asarray(client_sizes, dtype=np.int64)
        if sizes.sum() != n:
            raise ValueError(
                f"client_sizes 之和 ({sizes.sum()}) 必须等于样本数 ({n})"
            )

    proportions = rng.dirichlet(
        [label_alpha, label_alpha], size=n_clients
    )  # shape: (n_clients, 2)

    # alloc[i, c]：client i 分得的 class c 数量。逐类别分配，保证 client
    # 总收数恰好等于 sizes[i]。
    alloc = np.zeros((n_clients, 2), dtype=np.int64)
    remaining_cap = sizes.astype(np.int64).copy()
    class_labels = [0, 1]
    rng.shuffle(class_labels)  # 先分哪一类是任意的，不影响倾斜语义
    for step, c in enumerate(class_labels):
        n_class = int((y == c).sum())
        if step == len(class_labels) - 1:
            # 最后一个类别：填满所有剩余容量，保证无遗漏。
            alloc[:, c] = remaining_cap
            assert int(alloc[:, c].sum()) == n_class
            break
        # 偏好权重 = 容量 × Dirichlet 比例，只在仍有容量的 client 间分配。
        active = remaining_cap > 0
        weights = np.zeros(n_clients, dtype=np.float64)
        weights[active] = (
            remaining_cap[active].astype(np.float64) * proportions[active, c]
        )
        if weights.sum() <= 0:
            weights[active] = remaining_cap[active].astype(np.float64)
        counts = _largest_remainder(weights, n_class)
        # 不能超出任何 client 的剩余容量（权重 ∝ 容量时几乎不会发生，做一次裁剪兜底）。
        over = counts > remaining_cap
        if over.any():
            surplus = int((counts[over] - remaining_cap[over]).sum())
            counts[over] = remaining_cap[over]
            room = np.where(~over, remaining_cap - counts, 0)
            order = np.argsort(-room, kind="stable")
            for i in order:
                add = min(surplus, int(room[i]))
                counts[i] += add
                surplus -= add
                if surplus <= 0:
                    break
        alloc[:, c] = counts
        remaining_cap -= counts

    pools = {
        0: rng.permutation(np.where(y == 0)[0]),
        1: rng.permutation(np.where(y == 1)[0]),
    }
    offsets = {0: 0, 1: 0}
    splits: list[np.ndarray] = []
    for i in range(n_clients):
        idx_parts = []
        for c in (0, 1):
            take = int(alloc[i, c])
            idx_parts.append(pools[c][offsets[c] : offsets[c] + take])
            offsets[c] += take
        idx = np.concatenate(idx_parts)
        splits.append(np.sort(idx))
    return splits


def build_federated_dataset(
    n_clients: int = 5,
    n_samples: int = 2000,
    n_features: int = 10,
    iid: bool = False,
    label_alpha: float = 0.5,
    sizes_dirichlet: float | None = None,
    client_sizes: list[int] | np.ndarray | None = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    noise_std: float = 0.5,
) -> tuple[list[ClientData], np.ndarray, np.ndarray]:
    """生成合成数据并划分给多个本地 client。

    参数
    ----
    n_clients: client 数量。
    n_samples: 总样本数（划分前，含测试集部分）。
    n_features: 特征维度。
    iid: True 为 IID 同分布划分；False 为基于 Dirichlet 的 non-IID 标签倾斜。
    label_alpha: non-IID 时 Dirichlet 浓度，越小标签倾斜越严重
                 （例如 0.1 时某些 client 几乎只有单一标签）。
    sizes_dirichlet: 若非 None（如 0.3），再对 client 做一层数量倾斜：
                     各 client 规模 ∝ Dirichlet 抽样，小值意味着规模悬殊。
    client_sizes: 显式指定各 client 训练样本数（之和必须等于训练集大小），
                  与 sizes_dirichlet 二选一，用于精确测试“不同规模 client”。
    test_fraction: 统一留出测试集比例（全局测试集，不划分给任何 client）。
    seed: 随机种子，固定后数据生成与划分完全确定。

    返回
    ----
    clients: 每个 client 的 ClientData（仅训练数据）。
    X_test, y_test: 全局测试集。
    """
    if n_clients < 1:
        raise ValueError("n_clients 至少为 1")
    X, y = generate_classification_data(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
        noise_std=noise_std,
    )
    # 用独立的子流做训练/测试切分，保证调整 client 划分参数不影响测试集内容。
    ss = np.random.SeedSequence(seed)
    split_rng = np.random.default_rng(ss.spawn(1)[0])
    perm = split_rng.permutation(n_samples)
    n_test = int(round(n_samples * test_fraction))
    test_idx = perm[:n_test]
    train_idx = np.sort(perm[n_test:])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    part_rng = np.random.default_rng(ss.spawn(2)[0])
    n_train = y_train.shape[0]

    sizes: np.ndarray | None = None
    if client_sizes is not None:
        sizes = np.asarray(client_sizes, dtype=np.int64)
        if sizes.shape[0] != n_clients:
            raise ValueError("client_sizes 长度必须等于 n_clients")
        if sizes.sum() != n_train:
            raise ValueError(
                f"client_sizes 之和 ({sizes.sum()}) 必须等于训练样本数 ({n_train})"
            )
    elif sizes_dirichlet is not None:
        raw = part_rng.dirichlet([sizes_dirichlet] * n_clients)
        sizes = _largest_remainder(raw, n_train)

    if iid:
        index_splits = _iid_partition(y_train, n_clients, sizes, part_rng)
    else:
        index_splits = _non_iid_partition(
            y_train, n_clients, label_alpha, sizes, part_rng
        )

    clients = [
        ClientData(client_id=i, X=X_train[idx], y=y_train[idx])
        for i, idx in enumerate(index_splits)
    ]
    return clients, X_test, y_test

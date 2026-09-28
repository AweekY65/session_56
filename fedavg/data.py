"""合成数据生成与客户端划分（全部在本地由代码生成，不下载任何数据）。

合成任务：二维（或多维）二分类。先采样隐藏的真实权重 ``w_true``，
再生成特征 ``x ~ N(0, 1)``，标签 ``y ~ Bernoulli(sigmoid(x @ w_true + b))``，
从而逻辑回归在该数据上存在明确的最优解，FedAvg 能够收敛。

划分方式：
- IID：全局 shuffle 后按客户端样本数顺序切分，每个客户端标签分布近似一致。
- non-IID（标签偏斜）：用 Dirichlet(alpha) 在每个类别上按客户端抽取比例，
  alpha 越小，各客户端的标签分布越极端（部分客户端可能几乎只有一个类别）。
  若需要异构客户端规模，可传入 ``client_sizes``，按比例缩放划分结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Dataset:
    x: np.ndarray  # shape [n_samples, n_features]
    y: np.ndarray  # shape [n_samples]，取值 {0, 1}


@dataclass
class ClientData:
    client_id: int
    x: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def make_classification(
    n_samples: int = 2000,
    n_features: int = 6,
    seed: int = 42,
    w_scale: float = 1.5,
    w_true: Optional[np.ndarray] = None,
) -> Tuple[Dataset, np.ndarray]:
    """生成可分的二分类合成数据，返回 (数据集, 真实权重 w_true)。

    传入 ``w_true`` 时复用同一隐藏模型（训练集/测试集必须来自同一任务）。
    """
    rng = np.random.default_rng(seed)
    if w_true is None:
        w_true = rng.normal(scale=w_scale, size=n_features)
    w_true = np.asarray(w_true, dtype=float)
    x = rng.normal(size=(n_samples, n_features))
    logits = x @ w_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < probs).astype(np.int64)
    return Dataset(x=x, y=y), w_true


def _balanced_sizes(n_samples: int, n_clients: int) -> List[int]:
    base = n_samples // n_clients
    sizes = [base] * n_clients
    sizes[-1] += n_samples - base * n_clients
    return sizes


def iid_split(
    data: Dataset, n_clients: int, seed: int = 0,
    client_sizes: Optional[List[int]] = None,
) -> Dict[int, ClientData]:
    """IID 划分：全局打乱后顺序切分。"""
    rng = np.random.default_rng(seed)
    n = data.x.shape[0]
    if client_sizes is None:
        sizes = _balanced_sizes(n, n_clients)
    else:
        if len(client_sizes) != n_clients or sum(client_sizes) != n:
            raise ValueError("client_sizes 必须与客户端数一致且总和等于总样本数")
        sizes = list(client_sizes)
    perm = rng.permutation(n)
    clients: Dict[int, ClientData] = {}
    start = 0
    for cid, size in enumerate(sizes):
        idx = perm[start:start + size]
        start += size
        clients[cid] = ClientData(cid, data.x[idx], data.y[idx])
    return clients


def _dirichlet_label_split(
    data: Dataset, n_clients: int, alpha: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """按类别分别做 Dirichlet 比例分配，返回每个客户端的样本索引列表。"""
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]
    classes = np.unique(data.y)
    for cls in classes:
        cls_idx = np.where(data.y == cls)[0]
        rng.shuffle(cls_idx)
        # 比例 [n_clients]；切分点法保证该类所有样本都被分配。
        props = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
        cuts = np.cumsum(props)
        splits = np.split(cls_idx, (cuts[:-1] * len(cls_idx)).astype(int))
        for cid in range(n_clients):
            client_indices[cid].extend(splits[cid].tolist())
    return [np.array(ids, dtype=int) for ids in client_indices]


def non_iid_split(
    data: Dataset,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
    client_sizes: Optional[List[int]] = None,
) -> Dict[int, ClientData]:
    """non-IID 标签偏斜划分（Dirichlet）。

    参数 ``client_sizes`` 可同时制造规模异构：先做 Dirichlet 标签偏斜划分，
    再通过对每个客户端的索引子采样到目标规模（保持其标签偏斜比例近似不变）。
    """
    rng = np.random.default_rng(seed)
    n = data.x.shape[0]
    if client_sizes is not None:
        if len(client_sizes) != n_clients or sum(client_sizes) != n:
            raise ValueError("client_sizes 必须与客户端数一致且总和等于总样本数")

    indices = _dirichlet_label_split(data, n_clients, alpha, rng)

    if client_sizes is not None:
        # 规模异构：按目标大小对每客户端索引做有/无放回重采样。
        resampled: List[np.ndarray] = []
        for cid, target in enumerate(client_sizes):
            idx = indices[cid]
            if target == 0:
                raise ValueError(f"客户端 {cid} 的目标样本数为 0，无法训练")
            if len(idx) == 0:
                # 极端 alpha 下可能为空：从全局补一个样本，保证可训练。
                idx = rng.integers(0, n, size=1)
            replace = target > len(idx)
            chosen = rng.choice(idx, size=target, replace=replace)
            resampled.append(chosen)
        indices = resampled
    else:
        # 保证极端 alpha 下没有空客户端（空则从全局随机补样本）。
        for cid in range(n_clients):
            if len(indices[cid]) == 0:
                indices[cid] = rng.integers(0, n, size=1)

    clients: Dict[int, ClientData] = {}
    for cid, idx in enumerate(indices):
        rng.shuffle(idx)
        clients[cid] = ClientData(cid, data.x[idx], data.y[idx])
    return clients


def make_clients(
    n_clients: int = 5,
    n_samples: int = 2000,
    n_features: int = 6,
    iid: bool = True,
    alpha: float = 0.5,
    heterogeneous_sizes: bool = False,
    seed: int = 42,
) -> Tuple[Dict[int, ClientData], Dataset, Dataset, np.ndarray]:
    """便捷入口：生成数据并划分客户端。

    返回 ``(clients, train_data, test_data, w_true)``；训练集与测试集共享
    同一个隐藏权重 ``w_true``。
    """
    train, w_true = make_classification(n_samples, n_features, seed=seed)
    # 测试集必须与训练集来自同一隐藏模型，只换采样随机流。
    test, _ = make_classification(
        max(500, n_samples // 4), n_features, seed=seed + 1, w_true=w_true
    )

    split_rng_seed = seed + 1000
    sizes = None
    if heterogeneous_sizes:
        # 对数均匀的异构规模：小客户端与大客户端可相差数十倍。
        rng = np.random.default_rng(split_rng_seed)
        raw = rng.lognormal(mean=0.0, sigma=1.0, size=n_clients)
        raw /= raw.sum()
        sizes = np.floor(raw * n_samples).astype(int)
        # 修正取整误差，全部计入最后一个客户端。
        sizes[-1] += n_samples - int(sizes.sum())
        sizes = sizes.tolist()

    if iid:
        clients = iid_split(train, n_clients, seed=split_rng_seed, client_sizes=sizes)
    else:
        clients = non_iid_split(
            train, n_clients, alpha=alpha, seed=split_rng_seed, client_sizes=sizes
        )
    return clients, train, test, w_true

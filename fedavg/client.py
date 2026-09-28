"""FedAvg 聚合与客户端本地训练逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .data import ClientData
from .model import LogisticRegression


@dataclass
class ClientUpdate:
    """一次成功的本地训练结果。"""

    client_id: int
    params: Dict[str, np.ndarray]
    n_effective: int  # 有效样本数（实际参与训练的样本数），用于聚合加权


def federated_average(
    updates: List[ClientUpdate],
) -> Optional[Dict[str, np.ndarray]]:
    """按各客户端**有效样本数**加权的 FedAvg。

    .. math::
        \\theta_{t+1} = \\sum_{k} \\frac{n_k}{\\sum_j n_j}\\,\\theta_k^{(t)}

    - 空更新列表返回 ``None``（整轮无合法结果，调用方应跳过该轮）。
    - 权重严格归一化，避免简单平均不同规模客户端。
    - 含 NaN/Inf 的更新会被拒绝（见 :func:`valid_update`），调用方负责过滤。
    """
    if not updates:
        return None
    weights = np.array([float(u.n_effective) for u in updates], dtype=float)
    total = weights.sum()
    if total <= 0:
        return None
    weights = weights / total

    keys = list(updates[0].params.keys())
    agg: Dict[str, np.ndarray] = {}
    for key in keys:
        acc = np.zeros_like(updates[0].params[key], dtype=float)
        for w, u in zip(weights, updates):
            acc += w * u.params[key]
        agg[key] = acc
    return agg


def valid_update(update: ClientUpdate) -> bool:
    """更新必须有正的有效样本数且参数全部有限。"""
    if update.n_effective <= 0:
        return False
    return all(np.all(np.isfinite(v)) for v in update.params.values())


class Client:
    """一个联邦学习客户端：持有自己的本地数据，执行本地梯度下降。"""

    def __init__(
        self,
        client_id: int,
        data: ClientData,
        model: LogisticRegression,
        local_epochs: int = 2,
        lr: float = 0.1,
        drop_prob: float = 0.0,
        fail_prob: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.client_id = client_id
        self.data = data
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr
        self.drop_prob = drop_prob
        self.fail_prob = fail_prob
        # 全批量 GD 不使用随机数；rng 预留给未来的 mini-batch 扩展，
        # 同时用于掉线/失败随机判定（每个客户端独立流，互不干扰）。
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    def is_dropped(self) -> bool:
        """模拟通信掉线：客户端被选中但根本无法参与本轮。"""
        return self.drop_prob > 0 and self.rng.random() < self.drop_prob

    def local_train_fails(self) -> bool:
        """模拟本地训练失败（返回前崩溃 / 发散被检测）。"""
        return self.fail_prob > 0 and self.rng.random() < self.fail_prob

    def train(self, global_params: Dict[str, np.ndarray]) -> ClientUpdate:
        """执行本地训练。

        失败语义：
        - 掉线（:meth:`is_dropped`）或训练失败（:meth:`local_train_fails`）时
          抛出 :class:`ClientDroppedError` / :class:`ClientFailureError`，
          由模拟器捕获，该客户端不参与本轮聚合，其他客户端不受影响。
        - 正常完成时返回 :class:`ClientUpdate`，有效样本数即本地样本数。
        """
        if self.is_dropped():
            raise ClientDroppedError(self.client_id)
        if self.local_train_fails():
            raise ClientFailureError(self.client_id)
        if self.n_samples == 0:
            raise ClientFailureError(self.client_id, reason="empty dataset")

        new_params = self.model.train(
            global_params, self.data.x, self.data.y,
            lr=self.lr, epochs=self.local_epochs,
        )
        update = ClientUpdate(
            client_id=self.client_id,
            params=new_params,
            n_effective=self.n_samples,
        )
        if not valid_update(update):
            raise ClientFailureError(self.client_id, reason="non-finite update")
        return update

    def local_loss(self, params: Dict[str, np.ndarray]) -> float:
        return self.model.loss(params, self.data.x, self.data.y)


class ClientError(Exception):
    """客户端参与失败的基类（掉线/训练失败都不应打断整轮聚合）。"""


class ClientDroppedError(ClientError):
    def __init__(self, client_id: int, reason: str = "network drop"):
        super().__init__(f"client {client_id} dropped: {reason}")
        self.client_id = client_id
        self.reason = reason


class ClientFailureError(ClientError):
    def __init__(self, client_id: int, reason: str = "local training failed"):
        super().__init__(f"client {client_id} failed: {reason}")
        self.client_id = client_id
        self.reason = reason

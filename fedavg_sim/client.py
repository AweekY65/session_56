"""联邦学习本地 client。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ClientData
from .model import LocalTrainingError, Params, local_train


@dataclass
class ClientUpdate:
    """一个 client 一轮训练后提交的更新。

    effective_n_samples 是 FedAvg 聚合权重：client 本轮实际参与训练的
    有效样本数（默认等于本地数据集大小；训练失败时不产生本对象）。
    """

    client_id: int
    params: Params
    effective_n_samples: int
    train_loss: float

    def __post_init__(self) -> None:
        if self.effective_n_samples <= 0:
            raise ValueError("effective_n_samples 必须为正整数")
        if not np.isfinite(self.train_loss):
            raise ValueError("train_loss 必须是有限值")


class Client:
    """持有本地数据、执行本地训练的客户端。

    force_fail=True 时调用 train() 直接抛出 LocalTrainingError，
    用于模拟“本地训练失败”（例如进程崩溃、数据损坏）。
    """

    def __init__(
        self,
        data: ClientData,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        force_fail: bool = False,
    ) -> None:
        self.data = data
        self.client_id = data.client_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.force_fail = force_fail

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    def train(
        self, global_params: Params, *, round_idx: int, seed: int
    ) -> ClientUpdate:
        """从全局参数出发执行本地训练并返回更新。

        seed 由模拟器按 (round, client) 派生，保证可复现。
        失败时抛 LocalTrainingError，由模拟器捕获并跳过该 client。
        """
        if self.force_fail:
            raise LocalTrainingError(
                f"client {self.client_id} 本地训练失败（模拟故障）"
            )
        trained, loss = local_train(
            global_params,
            self.data.X,
            self.data.y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2=self.l2,
            seed=seed,
        )
        return ClientUpdate(
            client_id=self.client_id,
            params=trained,
            effective_n_samples=self.n_samples,
            train_loss=loss,
        )

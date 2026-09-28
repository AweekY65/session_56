"""轻量逻辑回归模型（纯 numpy 实现）。

模型：p(y=1|x) = sigmoid(x @ w + b)，使用 L2 正则的交叉熵损失，
通过全批量梯度下降进行本地训练（模拟器中每个客户端数据量很小，
全批量即可，且天然确定性、无采样随机性）。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # 数值稳定版 sigmoid，避免 z 绝对值很大时溢出。
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """二分类逻辑回归。

    参数以 ``{"w": ndarray[d], "b": ndarray[()]}`` 字典表示，
    方便 FedAvg 逐参数加权平均。
    """

    def __init__(self, n_features: int, l2: float = 0.0):
        self.n_features = int(n_features)
        self.l2 = float(l2)

    def init_params(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        # 小尺度随机初始化；初始权重不依赖任何客户端数据。
        w = rng.normal(scale=0.01, size=self.n_features)
        b = np.zeros(())
        return {"w": w.astype(float), "b": b.astype(float)}

    @staticmethod
    def probabilities(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ params["w"] + params["b"])

    def loss(self, params: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
        """平均交叉熵损失（含 L2 正则项）。"""
        p = self.probabilities(params, x)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * self.l2 * float(np.sum(params["w"] ** 2))
        return float(nll + reg)

    def accuracy(self, params: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
        pred = (self.probabilities(params, x) >= 0.5).astype(int)
        return float(np.mean(pred == y))

    def gradients(
        self, params: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        p = self.probabilities(params, x)
        err = p - y
        gw = (x.T @ err) / n + self.l2 * params["w"]
        gb = np.mean(err)
        return gw, gb

    def train(
        self,
        params: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        epochs: int,
    ) -> Dict[str, np.ndarray]:
        """全批量梯度下降，返回训练后的新参数（不原地修改输入）。"""
        w = params["w"].copy()
        b = params["b"].copy()
        for _ in range(int(epochs)):
            gw, gb = self.gradients({"w": w, "b": b}, x, y)
            w -= lr * gw
            b = b - lr * gb
        return {"w": w, "b": b}

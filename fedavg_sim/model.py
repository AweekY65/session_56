"""轻量逻辑回归模型与本地训练（纯 numpy，float64）。

模型: p = sigmoid(X @ w + b)
损失: 二元交叉熵（BCE），L2 正则可选
本地训练: mini-batch SGD，固定随机种子后逐批顺序完全确定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    """逻辑回归参数。所有计算与序列化均使用 float64。"""

    w: np.ndarray  # shape: (n_features,)
    b: float

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=np.float64).copy()
        self.b = float(self.b)

    @classmethod
    def zeros(cls, n_features: int) -> "Params":
        return cls(w=np.zeros(n_features, dtype=np.float64), b=0.0)

    def copy(self) -> "Params":
        return Params(w=self.w.copy(), b=self.b)

    def __eq__(self, other: object) -> bool:  # 仅供测试使用的严格比较
        if not isinstance(other, Params):
            return NotImplemented
        return bool(
            self.w.shape == other.w.shape
            and np.array_equal(self.w, other.w)
            and self.b == other.b
        )


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(params: Params, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return stable_sigmoid(X @ params.w + params.b)


def predict(params: Params, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(params, X) >= threshold).astype(np.float64)


def bce_loss(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> float:
    """在给定数据上计算平均二元交叉熵（含可选 L2 正则项 0.5*l2*||w||^2）。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    logits = X @ params.w + params.b
    # log(1+exp(z)) 的稳定形式：max(z,0) + log(1+exp(-|z|))
    loss = np.maximum(logits, 0.0) - y * logits + np.log1p(np.exp(-np.abs(logits)))
    total = float(loss.mean())
    if l2 > 0.0:
        total += 0.5 * l2 * float(params.w @ params.w)
    return total


def accuracy(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float((predict(params, X) == y).mean())


class LocalTrainingError(RuntimeError):
    """本地训练失败（数据无效、产生 NaN 等）。

    模拟器捕获该异常后会丢弃该 client 本轮的更新，不影响其他 client。
    """


def local_train(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    l2: float = 0.0,
    seed: int | None = None,
    check_finite: bool = True,
) -> tuple[Params, float]:
    """从全局参数 params 出发做本地 mini-batch SGD。

    返回 (训练后的参数, 训练后模型在本地数据上的平均 BCE loss)。
    seed 固定时，样本洗牌与更新顺序完全确定。

    可能抛出 LocalTrainingError：空数据、标签非法、训练后出现非有限值。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise LocalTrainingError("本地训练数据为空")
    if X.shape[0] != y.shape[0]:
        raise LocalTrainingError("X 与 y 样本数不一致")
    if not np.all((y == 0) | (y == 1)):
        raise LocalTrainingError("标签必须是 {0, 1}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise LocalTrainingError("训练数据包含非有限值")
    if not np.all(np.isfinite(params.w)) or not np.isfinite(params.b):
        raise LocalTrainingError("初始模型参数包含非有限值")

    new_params = params.copy()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    batch_size = min(batch_size, n)

    for _ in range(max(epochs, 1)):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X[idx], y[idx]
            probs = stable_sigmoid(xb @ new_params.w + new_params.b)
            error = probs - yb  # BCE 对 logits 的梯度
            m = xb.shape[0]
            grad_w = (xb.T @ error) / m
            if l2 > 0.0:
                grad_w = grad_w + l2 * new_params.w
            grad_b = float(error.mean())
            new_params.w -= learning_rate * grad_w
            new_params.b -= learning_rate * grad_b

    if check_finite and (
        not np.all(np.isfinite(new_params.w)) or not np.isfinite(new_params.b)
    ):
        raise LocalTrainingError("本地训练后参数出现 NaN/Inf")

    final_loss = bce_loss(new_params, X, y, l2=l2)
    if check_finite and not np.isfinite(final_loss):
        raise LocalTrainingError("本地训练后损失为 NaN/Inf")
    return new_params, final_loss

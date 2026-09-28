"""完全本地的 FedAvg 联邦学习模拟器（纯 numpy 实现）。

- data:       代码生成合成二分类数据，并按 IID / non-IID 划分给多个 client
- model:      轻量逻辑回归（float64）与本地 SGD 训练
- client:     本地客户端封装，可模拟训练失败
- aggregation:按有效样本数加权的 FedAvg
- simulation: 每轮采样参与者、处理掉线/失败、聚合、评估、checkpoint/恢复
"""
from .aggregation import (
    AggregatedModel,
    ClientUpdate,
    NoValidUpdatesError,
    fed_avg,
)
from .client import Client, LocalTrainingError
from .data import ClientData, build_federated_dataset
from .model import Params, accuracy, bce_loss, local_train, predict, predict_proba
from .simulation import FedAvgSimulator, SimConfig, RoundMetrics

__all__ = [
    "AggregatedModel",
    "Client",
    "ClientData",
    "ClientUpdate",
    "FedAvgSimulator",
    "LocalTrainingError",
    "NoValidUpdatesError",
    "Params",
    "RoundMetrics",
    "SimConfig",
    "accuracy",
    "bce_loss",
    "build_federated_dataset",
    "fed_avg",
    "local_train",
    "predict",
    "predict_proba",
]

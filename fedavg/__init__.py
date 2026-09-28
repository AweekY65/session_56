"""完全本地的 FedAvg 联邦学习模拟器（纯 numpy，无任何网络/外部服务依赖）。"""
from .client import (
    Client,
    ClientDroppedError,
    ClientFailureError,
    ClientUpdate,
    federated_average,
)
from .data import ClientData, Dataset, make_classification, make_clients
from .model import LogisticRegression
from .simulator import RoundRecord, SimConfig, Simulator

__all__ = [
    "Client",
    "ClientData",
    "ClientDroppedError",
    "ClientFailureError",
    "ClientUpdate",
    "Dataset",
    "LogisticRegression",
    "RoundRecord",
    "SimConfig",
    "Simulator",
    "federated_average",
    "make_classification",
    "make_clients",
]

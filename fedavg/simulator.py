"""FedAvg 联邦学习模拟器（完全本地、确定性、可 checkpoint 恢复）。

确定性设计
----------
给定同一个 ``seed``：

1. 数据生成与划分确定性复现（见 :mod:`fedavg.data`）。
2. 模型初始化使用从主 ``SeedSequence`` 派生的独立随机流。
3. 每轮参与者采样使用独立的 ``sampling_rng``，其 ``BitGenerator`` 状态
   会写入 checkpoint，恢复后采样序列逐轮一致。
4. 每个客户端每轮的掉线/失败判定使用由 ``(seed, round, client_id)``
   确定性派生的独立随机流——某个客户端失败不会“消耗”其他客户端的随机数，
   因此客户端执行顺序、个别失败都不会改变其余客户端的随机性。

失败语义
--------
- 掉线（被选中但无法通信）与本地训练失败都被模拟器捕获并记录，
  该客户端的更新被丢弃，**不参与**本轮聚合；
- 只要本轮至少有一个合法更新，聚合就按合法客户端的有效样本数加权进行；
- 若所有参与者都失败，本轮标记为 ``skipped``，全局参数不变，继续下一轮；
- 非有限值（NaN/Inf）的更新视为训练失败，同样被丢弃。

Checkpoint
----------
每轮结束原子写入（``tmp`` + ``os.replace``）一个 pickle 文件，包含配置、
轮次、全局参数、采样 RNG 状态和历史指标。``resume`` 时数据由相同 seed
确定性重建，最终模型与不中断运行在数值上一致（浮点运算顺序相同）。
"""
from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .client import (
    Client,
    ClientDroppedError,
    ClientError,
    ClientFailureError,
    ClientUpdate,
    federated_average,
)
from .data import make_clients
from .model import LogisticRegression

# 恢复训练时必须严格一致的配置项（决定数据与训练轨迹）；
# rounds（目标轮数）与 checkpoint_path（文件位置）允许在恢复时改变。
_STATEFUL_CONFIG_KEYS = (
    "seed", "n_clients", "n_samples", "n_features", "iid", "alpha",
    "heterogeneous_sizes", "participation_frac", "local_epochs", "lr",
    "l2", "drop_prob", "fail_prob",
)

CHECKPOINT_VERSION = 1


@dataclass
class SimConfig:
    """模拟器配置（全部可序列化，随 checkpoint 一起保存）。"""

    seed: int = 42
    rounds: int = 20
    n_clients: int = 5
    n_samples: int = 2000
    n_features: int = 6
    iid: bool = True
    alpha: float = 0.5  # non-IID Dirichlet 参数，越小越偏斜
    heterogeneous_sizes: bool = False
    participation_frac: float = 0.8  # 每轮参与客户端比例
    local_epochs: int = 2
    lr: float = 0.1
    l2: float = 0.0
    drop_prob: float = 0.0  # 每客户端每轮随机掉线概率
    fail_prob: float = 0.0  # 每客户端每轮随机训练失败概率
    checkpoint_path: Optional[str] = None

    def n_participants(self) -> int:
        return max(1, int(round(self.participation_frac * self.n_clients)))


@dataclass
class RoundRecord:
    round: int
    selected: List[int]
    succeeded: List[int]
    dropped: List[int]
    failed: List[int]
    n_effective_total: int
    loss: float
    accuracy: float
    skipped: bool = False


def _derive_seed(master_seed: int, *parts: int) -> int:
    """从 (master_seed, parts...) 确定性派生一个独立的随机种子。"""
    ss = np.random.SeedSequence([master_seed, *parts])
    return int(ss.generate_state(1)[0])


class Simulator:
    def __init__(
        self,
        config: SimConfig,
        failure_schedule: Optional[Dict[int, Dict[int, str]]] = None,
    ):
        """failure_schedule: ``{round: {client_id: "drop" | "fail"}}``，

        用于确定性地注入掉线/失败（测试与演示使用），与随机概率叠加。
        """
        self.config = config
        self.failure_schedule = failure_schedule or {}

        # 数据完全由代码确定性生成，保存在内存中；不访问任何外部资源。
        self.clients_data, self.train_data, self.test_data, _ = make_clients(
            n_clients=config.n_clients,
            n_samples=config.n_samples,
            n_features=config.n_features,
            iid=config.iid,
            alpha=config.alpha,
            heterogeneous_sizes=config.heterogeneous_sizes,
            seed=config.seed,
        )
        self.model = LogisticRegression(config.n_features, l2=config.l2)
        self.clients: Dict[int, Client] = {}
        for cid, cdata in self.clients_data.items():
            self.clients[cid] = Client(
                client_id=cid,
                data=cdata,
                model=self.model,
                local_epochs=config.local_epochs,
                lr=config.lr,
                drop_prob=config.drop_prob,
                fail_prob=config.fail_prob,
            )

        # 独立的初始化流与采样流。
        init_rng = np.random.default_rng(_derive_seed(config.seed, 0))
        self.global_params = self.model.init_params(init_rng)
        self.sampling_rng = np.random.default_rng(_derive_seed(config.seed, 1))

        self.round = 0  # 下一个要执行的轮次
        self.history: List[RoundRecord] = []

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "round": self.round,
            "global_params": self.global_params,
            "sampling_rng_state": self.sampling_rng.bit_generator.state,
            "history": [asdict(r) for r in self.history],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"不兼容的 checkpoint 版本: {state.get('version')}"
            )
        saved_cfg = state["config"]
        current_cfg = asdict(self.config)
        mismatched = [
            k for k in _STATEFUL_CONFIG_KEYS if saved_cfg.get(k) != current_cfg.get(k)
        ]
        if mismatched:
            raise ValueError(
                "checkpoint 与当前模拟器的关键配置不一致: "
                + ", ".join(f"{k}: {saved_cfg.get(k)} != {current_cfg.get(k)}" for k in mismatched)
            )
        self.round = int(state["round"])
        self.global_params = {
            k: np.asarray(v) for k, v in state["global_params"].items()
        }
        self.sampling_rng.bit_generator.state = state["sampling_rng_state"]
        self.history = [RoundRecord(**r) for r in state["history"]]

    def save_checkpoint(self, path: str) -> None:
        # 原子写：先写临时文件再 rename，避免中途崩溃产生损坏 checkpoint。
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.state_dict(), f, protocol=5)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def resume(
        cls,
        path: str,
        config_override: Optional[SimConfig] = None,
        failure_schedule: Optional[Dict[int, Dict[int, str]]] = None,
    ) -> "Simulator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        config = config_override or SimConfig(**state["config"])
        sim = cls(config, failure_schedule=failure_schedule)
        sim.load_state_dict(state)
        return sim

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    def _client_behavior_rng(self, round_idx: int, client_id: int):
        # 每个 (轮次, 客户端) 独立确定的随机流，互不影响。
        return np.random.default_rng(
            _derive_seed(self.config.seed, 100_000 + round_idx, client_id)
        )

    def _run_round(self, round_idx: int) -> RoundRecord:
        cfg = self.config
        all_ids = sorted(self.clients.keys())
        m = min(cfg.n_participants(), len(all_ids))
        selected = self.sampling_rng.choice(
            all_ids, size=m, replace=False
        ).tolist()

        succeeded: List[int] = []
        dropped: List[int] = []
        failed: List[int] = []
        updates: List[ClientUpdate] = []

        for cid in selected:
            client = self.clients[cid]
            forced = self.failure_schedule.get(round_idx, {}).get(cid)
            try:
                if forced == "drop":
                    raise ClientDroppedError(cid, reason="forced drop")
                if forced == "fail":
                    raise ClientFailureError(cid, reason="forced failure")
                # 为本轮注入确定性的独立行为 RNG（掉线/失败判定）。
                client.rng = self._client_behavior_rng(round_idx, cid)
                update = client.train(self.global_params)
            except ClientDroppedError:
                dropped.append(cid)
                continue
            except ClientFailureError:
                failed.append(cid)
                continue
            except ClientError:  # 防御：任何客户端异常都不得打断整轮聚合
                failed.append(cid)
                continue

            updates.append(update)
            succeeded.append(cid)

        agg = federated_average(updates)
        skipped = agg is None
        if not skipped:
            self.global_params = agg

        loss = self.model.loss(self.global_params, self.test_data.x, self.test_data.y)
        acc = self.model.accuracy(
            self.global_params, self.test_data.x, self.test_data.y
        )
        n_eff = sum(u.n_effective for u in updates)
        return RoundRecord(
            round=round_idx,
            selected=selected,
            succeeded=succeeded,
            dropped=dropped,
            failed=failed,
            n_effective_total=n_eff,
            loss=loss,
            accuracy=acc,
            skipped=skipped,
        )

    def run(self, verbose: bool = True, checkpoint_every: int = 1) -> List[RoundRecord]:
        """执行从当前轮次到 ``config.rounds`` 的所有轮次。"""
        cfg = self.config
        while self.round < cfg.rounds:
            r = self.round
            record = self._run_round(r)
            self.history.append(record)
            self.round = r + 1

            if verbose:
                status = "SKIP" if record.skipped else "OK  "
                print(
                    f"[round {r:3d}] {status} "
                    f"selected={record.selected} "
                    f"ok={record.succeeded} drop={record.dropped} fail={record.failed} "
                    f"n_eff={record.n_effective_total} "
                    f"loss={record.loss:.4f} acc={record.accuracy:.4f}"
                )

            if cfg.checkpoint_path and checkpoint_every > 0:
                if self.round % checkpoint_every == 0 or self.round >= cfg.rounds:
                    self.save_checkpoint(cfg.checkpoint_path)

        if verbose:
            final = self.history[-1]
            print(
                f"训练结束：共 {len(self.history)} 轮，"
                f"最终 loss={final.loss:.4f} accuracy={final.accuracy:.4f}"
            )
        return self.history

    # ------------------------------------------------------------------
    # 便捷指标
    # ------------------------------------------------------------------
    def evaluate(self) -> Tuple[float, float]:
        loss = self.model.loss(self.global_params, self.test_data.x, self.test_data.y)
        acc = self.model.accuracy(
            self.global_params, self.test_data.x, self.test_data.y
        )
        return loss, acc

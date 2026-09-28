"""联邦训练主循环：采样、掉线/失败处理、FedAvg、评估、checkpoint。

确定性
------
所有随机行为（参与者采样、掉线、训练失败、本地 mini-batch 洗牌）都通过
numpy SeedSequence 从 ``config.seed`` 按“轮次 + 用途”派生子随机流：

    root = SeedSequence(seed)
    round_seed = root.spawn(rounds)[round_idx]
    sample/dropout/failure/train 四个子流 = round_seed.spawn(4)

因此恢复训练时不需要保存/恢复任何 RNG 内部状态——只要轮次相同，
派生出的随机流就完全相同。checkpoint 里只需要保存轮次与模型参数。

失败语义
--------
- 掉线 (dropout)：被采样但本轮无法联系，不执行本地训练；
- 训练失败 (failure)：执行训练时抛出 LocalTrainingError（含模拟故障、
  数据问题、NaN），其更新被丢弃；
- 单个/部分 client 失败不影响本轮其他 client 的合法聚合；
- 当成功 client 数 < min_successful_participants（默认 1）或聚合器发现
  所有更新都非法时，本轮跳过（全局模型保持不变），训练继续进行。
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .aggregation import NoValidUpdatesError, fed_avg
from .client import Client, ClientUpdate, LocalTrainingError
from .data import ClientData
from .model import Params, accuracy, bce_loss


@dataclass(frozen=True)
class SimConfig:
    """联邦训练配置。该对象完整刻画运行行为，可 JSON 序列化进 checkpoint。"""

    seed: int = 42
    rounds: int = 20
    n_participants: int | None = None  # 每轮采样数；None 表示全部 client
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.1
    l2: float = 0.0
    dropout_prob: float = 0.0          # 每个被采样 client 的掉线概率
    failure_prob: float = 0.0          # 每个在线 client 的本地训练失败概率
    offline_client_ids: tuple[int, ...] = ()   # 永久离线，从不参与采样
    always_fail_client_ids: tuple[int, ...] = ()  # 一旦被采样必训练失败
    min_successful_participants: int = 1
    checkpoint_every: int = 1          # 每 N 轮保存一次 checkpoint


@dataclass
class RoundMetrics:
    round: int
    sampled_ids: tuple[int, ...]
    dropped_ids: tuple[int, ...]
    failed_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    contributing_ids: tuple[int, ...]
    n_successful: int
    total_effective_samples: int
    train_loss: float | None           # 按有效样本数加权的本地训练损失
    test_loss: float | None
    test_accuracy: float | None
    skipped: bool                      # True 表示本轮无合法聚合，模型未更新


@dataclass
class _DatasetMeta:
    n_clients: int
    n_features: int
    client_sizes: tuple[int, ...]
    n_test: int


class FedAvgSimulator:
    """FedAvg 联邦学习模拟器（完全本地、无任何网络/外部服务调用）。"""

    def __init__(
        self,
        clients_data: list[ClientData],
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: SimConfig,
        *,
        checkpoint_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if not clients_data:
            raise ValueError("至少需要一个 client")
        self.clients_data = clients_data
        self.X_test = np.asarray(X_test, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        n_features = clients_data[0].X.shape[1]
        for cd in clients_data:
            if cd.X.shape[1] != n_features:
                raise ValueError("所有 client 的特征维度必须一致")
        if self.X_test.shape[1] != n_features:
            raise ValueError("测试集特征维度与 client 数据不一致")
        self._n_features = n_features

        n_clients = len(clients_data)
        k = config.n_participants if config.n_participants is not None else n_clients
        self._k = min(max(k, 1), n_clients)
        offline = set(config.offline_client_ids)
        self._eligible = [
            cd.client_id for cd in clients_data if cd.client_id not in offline
        ]
        if not self._eligible:
            raise ValueError("没有任何可参与的 client（全部永久离线）")

        self._clients: dict[int, Client] = {
            cd.client_id: Client(
                cd,
                epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                l2=config.l2,
                force_fail=cd.client_id in set(config.always_fail_client_ids),
            )
            for cd in clients_data
        }

        self.global_params: Params = Params.zeros(n_features)
        self.next_round: int = 1
        self.history: list[RoundMetrics] = []
        # root.spawn(rounds) 为每一轮预生成独立 SeedSequence。
        self._round_seeds = np.random.SeedSequence(config.seed).spawn(config.rounds)

    # ------------------------------------------------------------------ #
    # 核心循环
    # ------------------------------------------------------------------ #
    def _round_rngs(self, round_idx: int):
        sample_ss, dropout_ss, failure_ss, train_ss = self._round_seeds[
            round_idx - 1
        ].spawn(4)
        return (
            np.random.default_rng(sample_ss),
            np.random.default_rng(dropout_ss),
            np.random.default_rng(failure_ss),
            train_ss,
        )

    def _select_participants(self, rng: np.random.Generator) -> list[int]:
        perm = rng.permutation(np.asarray(self._eligible))
        return [int(cid) for cid in perm[: self._k]]

    def _run_round(self, round_idx: int) -> RoundMetrics:
        sample_rng, dropout_rng, failure_rng, train_ss = self._round_rngs(round_idx)

        sampled = self._select_participants(sample_rng)

        # 1) 掉线：本轮不执行训练（按被采样位置消耗随机数）。
        drop_draws = dropout_rng.random(len(sampled))
        dropped_set = {
            cid for cid, u in zip(sampled, drop_draws) if u < self.config.dropout_prob
        }
        dropped = [cid for cid in sampled if cid in dropped_set]

        # 2) 训练失败：先按概率确定随机故障，再实际训练（异常同样视为失败）。
        # 随机数同样按被采样位置消耗；掉线 client 不会被判为失败。
        fail_draws = failure_rng.random(len(sampled))
        will_fail_stochastic = {
            cid
            for cid, u in zip(sampled, fail_draws)
            if cid not in dropped_set and u < self.config.failure_prob
        }

        # 每个被采样位置一个训练种子：掉线/失败 client 也消耗自己的位置，
        # 保证其他 client 的本地训练种子与谁掉线/失败完全无关。
        train_seed_ss = train_ss.spawn(len(sampled))
        updates: list[ClientUpdate] = []
        failed: list[int] = []
        for pos, cid in enumerate(sampled):
            if cid in dropped_set:
                continue
            seed_for_client = train_seed_ss[pos]
            if cid in will_fail_stochastic:
                failed.append(cid)
                continue
            try:
                upd = self._clients[cid].train(
                    self.global_params,
                    round_idx=round_idx,
                    seed=int(seed_for_client.generate_state(1)[0]),
                )
            except LocalTrainingError:
                failed.append(cid)
                continue
            updates.append(upd)

        # 3) 按有效样本数加权聚合；失败 client 不影响其余 client。
        try:
            agg = fed_avg(updates)
        except NoValidUpdatesError:
            agg = None

        # 成功 client 数低于下限时同样跳过本轮（模型保持不变）。
        if agg is not None and len(agg.contributing_client_ids) < (
            self.config.min_successful_participants
        ):
            agg = None

        if agg is None:
            test_loss = bce_loss(self.global_params, self.X_test, self.y_test)
            test_acc = accuracy(self.global_params, self.X_test, self.y_test)
            return RoundMetrics(
                round=round_idx,
                sampled_ids=tuple(sampled),
                dropped_ids=tuple(dropped),
                failed_ids=tuple(failed),
                rejected_ids=tuple(),
                contributing_ids=(),
                n_successful=0,
                total_effective_samples=0,
                train_loss=None,
                test_loss=test_loss,
                test_accuracy=test_acc,
                skipped=True,
            )

        by_id = {u.client_id: u for u in updates}
        train_loss = (
            sum(
                by_id[cid].effective_n_samples * by_id[cid].train_loss
                for cid in agg.contributing_client_ids
            )
            / agg.total_effective_samples
        )
        self.global_params = agg.params

        test_loss = bce_loss(self.global_params, self.X_test, self.y_test)
        test_acc = accuracy(self.global_params, self.X_test, self.y_test)
        return RoundMetrics(
            round=round_idx,
            sampled_ids=tuple(sampled),
            dropped_ids=tuple(dropped),
            failed_ids=tuple(failed),
            rejected_ids=agg.rejected_client_ids,
            contributing_ids=agg.contributing_client_ids,
            n_successful=len(agg.contributing_client_ids),
            total_effective_samples=agg.total_effective_samples,
            train_loss=float(train_loss),
            test_loss=float(test_loss),
            test_accuracy=float(test_acc),
            skipped=False,
        )

    def run(self, stop_after_round: int | None = None) -> list[RoundMetrics]:
        """执行训练循环。

        stop_after_round: 跑到第几轮停止（含），None 表示跑满 config.rounds。
        典型 checkpoint 用法：先 run(k)，重建模拟器 resume 后再 run(None)。
        """
        last = self.config.rounds if stop_after_round is None else min(stop_after_round, self.config.rounds)
        while self.next_round <= last:
            metrics = self._run_round(self.next_round)
            self.history.append(metrics)
            self.next_round += 1
            if self.checkpoint_dir is not None and (
                self.config.checkpoint_every > 0
                and (
                    (metrics.round % self.config.checkpoint_every == 0)
                    or metrics.round == self.config.rounds
                )
            ):
                self.save_checkpoint(self.checkpoint_dir)
        return self.history

    # ------------------------------------------------------------------ #
    # Checkpoint（仅本地文件；临时文件 + 原子 rename）
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> Path:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)

        state = {
            "next_round": self.next_round,
            "history": [self._metrics_to_dict(m) for m in self.history],
            "data_meta": asdict(
                _DatasetMeta(
                    n_clients=len(self.clients_data),
                    n_features=self._n_features,
                    client_sizes=tuple(cd.n_samples for cd in self.clients_data),
                    n_test=int(self.y_test.shape[0]),
                )
            ),
            "version": 1,
        }
        self._atomic_write_json(ckpt / "config.json", self._config_to_dict(self.config))
        self._atomic_write_json(ckpt / "state.json", state)
        self._atomic_save_npz(
            ckpt / "model.npz", w=self.global_params.w, b=np.asarray(self.global_params.b)
        )
        return ckpt

    @classmethod
    def resume(
        cls,
        checkpoint_dir: str | os.PathLike[str],
        clients_data: list[ClientData],
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        config: SimConfig | None = None,
    ) -> "FedAvgSimulator":
        """从 checkpoint 恢复。传入的 config 必须与保存时一致（默认从目录读取）。"""
        ckpt = Path(checkpoint_dir)
        saved_config = cls._config_from_dict(json.loads((ckpt / "config.json").read_text()))
        if config is not None and cls._config_to_dict(config) != cls._config_to_dict(saved_config):
            raise ValueError("恢复时传入的 config 与 checkpoint 中保存的不一致")

        state = json.loads((ckpt / "state.json").read_text())
        sim = cls(clients_data, X_test, y_test, saved_config, checkpoint_dir=ckpt)

        meta = state["data_meta"]
        if meta["n_clients"] != len(clients_data):
            raise ValueError("client 数量与 checkpoint 不一致")
        if meta["n_features"] != sim._n_features:
            raise ValueError("特征维度与 checkpoint 不一致")
        if tuple(meta["client_sizes"]) != tuple(cd.n_samples for cd in clients_data):
            raise ValueError("各 client 样本数与 checkpoint 不一致——请用相同 seed 重新生成数据")
        if meta["n_test"] != int(y_test.shape[0]):
            raise ValueError("测试集大小与 checkpoint 不一致")

        model = np.load(ckpt / "model.npz")
        sim.global_params = Params(w=model["w"].astype(np.float64), b=float(model["b"]))
        sim.next_round = int(state["next_round"])
        sim.history = [cls._metrics_from_dict(m) for m in state["history"]]
        return sim

    # ------------------------------------------------------------------ #
    # 序列化辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp.npz"
        )
        os.close(fd)
        try:
            np.savez(tmp, **arrays)  # tmp 已以 .npz 结尾，numpy 不会再追加扩展名
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _config_to_dict(cfg: SimConfig) -> dict:
        d = asdict(cfg)
        d["offline_client_ids"] = list(d["offline_client_ids"])
        d["always_fail_client_ids"] = list(d["always_fail_client_ids"])
        return d

    @staticmethod
    def _config_from_dict(d: dict) -> SimConfig:
        return SimConfig(
            seed=int(d["seed"]),
            rounds=int(d["rounds"]),
            n_participants=(None if d["n_participants"] is None else int(d["n_participants"])),
            local_epochs=int(d["local_epochs"]),
            batch_size=int(d["batch_size"]),
            learning_rate=float(d["learning_rate"]),
            l2=float(d["l2"]),
            dropout_prob=float(d["dropout_prob"]),
            failure_prob=float(d["failure_prob"]),
            offline_client_ids=tuple(int(x) for x in d["offline_client_ids"]),
            always_fail_client_ids=tuple(int(x) for x in d["always_fail_client_ids"]),
            min_successful_participants=int(d["min_successful_participants"]),
            checkpoint_every=int(d["checkpoint_every"]),
        )

    @staticmethod
    def _metrics_to_dict(m: RoundMetrics) -> dict:
        return {
            "round": m.round,
            "sampled_ids": list(m.sampled_ids),
            "dropped_ids": list(m.dropped_ids),
            "failed_ids": list(m.failed_ids),
            "rejected_ids": list(m.rejected_ids),
            "contributing_ids": list(m.contributing_ids),
            "n_successful": m.n_successful,
            "total_effective_samples": m.total_effective_samples,
            "train_loss": m.train_loss,
            "test_loss": m.test_loss,
            "test_accuracy": m.test_accuracy,
            "skipped": m.skipped,
        }

    @staticmethod
    def _metrics_from_dict(d: dict) -> RoundMetrics:
        tup = lambda k: tuple(int(x) for x in d[k])  # noqa: E731
        return RoundMetrics(
            round=int(d["round"]),
            sampled_ids=tup("sampled_ids"),
            dropped_ids=tup("dropped_ids"),
            failed_ids=tup("failed_ids"),
            rejected_ids=tup("rejected_ids"),
            contributing_ids=tup("contributing_ids"),
            n_successful=int(d["n_successful"]),
            total_effective_samples=int(d["total_effective_samples"]),
            train_loss=d["train_loss"],
            test_loss=d["test_loss"],
            test_accuracy=d["test_accuracy"],
            skipped=bool(d["skipped"]),
        )

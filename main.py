"""终端 demo：在完全本地的合成数据上运行 FedAvg 联邦学习。

示例：
    python main.py                                   # 默认 non-IID，20 轮
    python main.py --iid --rounds 30 --seed 1
    python main.py --n-clients 8 --participants 4 \
        --dropout-prob 0.2 --failure-prob 0.1 \
        --sizes-dirichlet 0.3 --checkpoint-dir ./ckpt_demo
    python main.py --resume ./ckpt_demo             # 从 checkpoint 恢复

所有数据由代码生成，全程无任何网络/外部服务调用。
"""
from __future__ import annotations

import argparse

import numpy as np

from fedavg_sim import FedAvgSimulator, SimConfig, build_federated_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="完全本地的 FedAvg 联邦学习模拟器")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--rounds", type=int, default=20, help="全局通信轮数")
    p.add_argument("--n-clients", type=int, default=5, help="client 总数")
    p.add_argument("--participants", type=int, default=3, help="每轮采样 client 数")
    p.add_argument("--n-samples", type=int, default=2000, help="合成样本总数")
    p.add_argument("--n-features", type=int, default=10, help="特征维度")
    p.add_argument("--local-epochs", type=int, default=2, help="每轮本地 epoch 数")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1, help="本地学习率")
    p.add_argument("--iid", action="store_true", help="使用 IID 数据划分（默认 non-IID）")
    p.add_argument("--label-alpha", type=float, default=0.5,
                   help="non-IID Dirichlet 浓度，越小越倾斜")
    p.add_argument("--sizes-dirichlet", type=float, default=None,
                   help="数量倾斜 Dirichlet 浓度（如 0.3 表示 client 规模悬殊）")
    p.add_argument("--dropout-prob", type=float, default=0.0, help="每 client 掉线概率")
    p.add_argument("--failure-prob", type=float, default=0.0, help="每 client 训练失败概率")
    p.add_argument("--offline-ids", type=int, nargs="*", default=[], help="永久离线 client id")
    p.add_argument("--always-fail-ids", type=int, nargs="*", default=[],
                   help="一旦被采样就训练失败的 client id")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint 目录")
    p.add_argument("--resume", type=str, default=None,
                   help="从指定 checkpoint 目录恢复并继续训练")
    return p.parse_args()


def build_data(args: argparse.Namespace):
    return build_federated_dataset(
        n_clients=args.n_clients,
        n_samples=args.n_samples,
        n_features=args.n_features,
        iid=args.iid,
        label_alpha=args.label_alpha,
        sizes_dirichlet=args.sizes_dirichlet,
        seed=args.seed,
    )


def print_header(clients, iid: bool) -> None:
    print("=" * 72)
    print("本地 FedAvg 联邦学习模拟器（合成二分类数据，逻辑回归）")
    print(f"划分方式: {'IID' if iid else 'non-IID (Dirichlet 标签倾斜)'}")
    print(f"各 client 训练样本数: {[c.n_samples for c in clients]}")
    print("=" * 72)


def print_round(m) -> None:
    status = "SKIP" if m.skipped else "ok  "
    print(
        f"round {m.round:02d} [{status}] "
        f"采样={list(m.sampled_ids)} 掉线={list(m.dropped_ids)} "
        f"失败={list(m.failed_ids)} 贡献={list(m.contributing_ids)} "
        f"有效样本={m.total_effective_samples:5d} "
        f"train_loss={m.train_loss if m.train_loss is not None else float('nan'):.4f} "
        f"test_loss={m.test_loss:.4f} acc={m.test_accuracy:.4f}"
    )


def main() -> None:
    args = parse_args()
    clients_data, X_test, y_test = build_data(args)

    if args.resume:
        sim = FedAvgSimulator.resume(
            args.resume,
            clients_data,
            X_test,
            y_test,
        )
        print(f"已从 {args.resume} 恢复，将从第 {sim.next_round} 轮继续")
    else:
        config = SimConfig(
            seed=args.seed,
            rounds=args.rounds,
            n_participants=args.participants,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dropout_prob=args.dropout_prob,
            failure_prob=args.failure_prob,
            offline_client_ids=tuple(args.offline_ids),
            always_fail_client_ids=tuple(args.always_fail_ids),
        )
        sim = FedAvgSimulator(
            clients_data, X_test, y_test, config,
            checkpoint_dir=args.checkpoint_dir,
        )
        print_header(clients_data, args.iid)

    history = sim.run()
    for m in history:
        print_round(m)

    valid = [m for m in history if not m.skipped]
    print("-" * 72)
    if valid:
        last = valid[-1]
        print(
            f"训练完成: 共 {len(history)} 轮（跳过 {len(history) - len(valid)} 轮），"
            f"最终 test_loss={last.test_loss:.4f}, test_accuracy={last.test_accuracy:.4f}"
        )
        print(f"全局模型 w 的 L2 范数: {np.linalg.norm(sim.global_params.w):.4f}, "
              f"b={sim.global_params.b:.4f}")
    else:
        print("训练完成，但所有轮次均被跳过（无任何成功 client）。")


if __name__ == "__main__":
    main()

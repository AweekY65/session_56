"""命令行入口：本地运行一次 FedAvg 模拟（或从 checkpoint 恢复）。

示例
----
IID 训练::

    python -m fedavg.main --iid --rounds 20 --seed 42

non-IID（Dirichlet 标签偏斜）+ 随机掉线/失败 + checkpoint::

    python -m fedavg.main --no-iid --alpha 0.3 --drop-prob 0.1 \\
        --fail-prob 0.1 --checkpoint ckpt.pkl

从 checkpoint 恢复::

    python -m fedavg.main --resume --checkpoint ckpt.pkl --rounds 30
"""
from __future__ import annotations

import argparse

from .simulator import SimConfig, Simulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 FedAvg 联邦学习模拟器")
    p.add_argument("--seed", type=int, default=42, help="固定随机种子")
    p.add_argument("--rounds", type=int, default=20, help="联邦轮数")
    p.add_argument("--clients", type=int, default=5, help="客户端总数")
    p.add_argument("--samples", type=int, default=2000, help="合成训练样本数")
    p.add_argument("--features", type=int, default=6, help="特征维度")
    p.add_argument("--iid", dest="iid", action="store_true", default=True)
    p.add_argument("--no-iid", dest="iid", action="store_false", help="使用 non-IID 划分")
    p.add_argument("--alpha", type=float, default=0.5, help="non-IID Dirichlet 浓度参数")
    p.add_argument(
        "--heterogeneous-sizes", action="store_true",
        help="启用异构客户端规模（对数正态采样）",
    )
    p.add_argument("--participation", type=float, default=0.8, help="每轮参与客户端比例")
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.5, help="本地学习率")
    p.add_argument("--l2", type=float, default=0.0, help="L2 正则系数")
    p.add_argument("--drop-prob", type=float, default=0.0, help="每客户端每轮掉线概率")
    p.add_argument("--fail-prob", type=float, default=0.0, help="每客户端每轮训练失败概率")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint 文件路径")
    p.add_argument("--resume", action="store_true", help="从 checkpoint 恢复继续训练")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        seed=args.seed,
        rounds=args.rounds,
        n_clients=args.clients,
        n_samples=args.samples,
        n_features=args.features,
        iid=args.iid,
        alpha=args.alpha,
        heterogeneous_sizes=args.heterogeneous_sizes,
        participation_frac=args.participation,
        local_epochs=args.local_epochs,
        lr=args.lr,
        l2=args.l2,
        drop_prob=args.drop_prob,
        fail_prob=args.fail_prob,
        checkpoint_path=args.checkpoint,
    )


def main() -> None:
    args = parse_args()
    if args.resume:
        if not args.checkpoint:
            raise SystemExit("--resume 需要同时指定 --checkpoint 路径")
        config = build_config(args)
        sim = Simulator.resume(args.checkpoint, config_override=config)
        print(f"已从 {args.checkpoint} 恢复：从第 {sim.round} 轮继续（目标 {config.rounds} 轮）")
    else:
        sim = Simulator(build_config(args))
        dist = "IID" if args.iid else f"non-IID(alpha={args.alpha})"
        print(
            f"本地 FedAvg 模拟：{args.clients} 个客户端，{args.samples} 个合成样本，"
            f"{dist} 划分，参与比例 {args.participation}，seed={args.seed}"
        )
    sim.run(verbose=True)
    loss, acc = sim.evaluate()
    print(f"最终测试集指标：loss={loss:.4f} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()

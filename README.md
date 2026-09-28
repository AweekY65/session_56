# 本地 FedAvg 联邦学习模拟器

一个**完全本地**的联邦学习模拟器：使用轻量逻辑回归 + FedAvg，在多个本地
客户端上对**代码生成的合成数据**进行联邦训练。纯 numpy 实现，不连接任何
模型服务、云训练平台、数据库或外部服务，不下载任何数据集或预训练模型。

## 目录结构

```
fedavg/
├── __init__.py     # 包导出
├── data.py         # 合成数据生成与 IID / non-IID 客户端划分
├── model.py        # 逻辑回归（数值稳定 sigmoid、交叉熵、全批量 GD）
├── client.py       # 客户端本地训练、掉线/失败语义、加权 FedAvg 聚合
├── simulator.py    # 模拟器主循环、确定性随机流、checkpoint 存取
└── main.py         # 命令行入口
tests/              # 31 个 pytest 用例（IID/non-IID、规模异构、掉线、
                    # 失败、聚合权重、checkpoint、确定性、本地化约束）
```

## FedAvg 流程

1. **数据生成（本地）**：采样一个隐藏权重 `w_true`，生成
   `x ~ N(0, I)`，标签 `y ~ Bernoulli(sigmoid(x @ w_true))`。
   训练集与测试集共享同一个 `w_true`（同一任务，不同采样流）。
2. **客户端划分**：把全部训练样本划分给 `K` 个客户端（IID 或 non-IID）。
3. **每一轮 `t`**：
   1. 服务器按参与比例随机抽取 `m = max(1, round(C·K))` 个客户端；
   2. 把当前全局参数 `θ_t` 下发给被选中的客户端；
   3. 各客户端在**本地数据**上做全批量梯度下降（`E` 个 local epoch），
      得到 `θ_k^(t+1)`；
   4. 服务器收集成功客户端的更新，**按有效样本数加权聚合**：

   $$\theta_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j}\, \theta_k^{(t+1)}$$

   注意：权重只来自本轮**成功**的客户端，且按其真实有效样本数 `n_k`
   归一化——数据量 300 的客户端权重是 100 的客户端的 3 倍，绝不是简单平均。
4. 在独立测试集上评估 loss / accuracy，进入下一轮。

## 数据生成方式

- **IID**：全局 shuffle 后按客户端样本数顺序切分，各客户端标签分布近似一致。
- **non-IID（标签偏斜）**：对每个类别独立采样 `Dirichlet(α)` 比例分配给
  各客户端。`α` 越小偏斜越强（如 `α=0.05` 时部分客户端几乎只有单一类别）。
- **规模异构**：`--heterogeneous-sizes` 用对数正态分布生成各客户端样本数，
  大/小客户端可相差数十倍（此时加权聚合尤其重要）。

所有随机性都来自固定 `seed` 派生的 numpy `Generator`，数据只存在于内存。

## 失败语义

模拟两类客户端异常，均**不破坏整轮合法聚合**：

| 情况 | 行为 |
|---|---|
| 通信掉线（`drop`） | 客户端不返回更新，记录到 `dropped`，不参与聚合 |
| 本地训练失败（`fail`） | 抛出异常被模拟器捕获，记录到 `failed`，更新被丢弃 |
| 更新含 NaN/Inf | 视为训练失败，更新被丢弃 |
| 本轮所有参与者都失败 | 该轮标记 `skipped`，**全局参数保持不变**，继续下一轮 |

失败注入有两种方式：随机概率（`--drop-prob` / `--fail-prob`，按
`(seed, round, client_id)` 确定性派生独立随机流），以及确定性的
`failure_schedule={round: {client_id: "drop"|"fail"}}`（测试用）。
单个客户端失败不会消耗或影响其他客户端的随机流。

## 确定性

固定 `seed` 时以下全部可复现：数据生成与划分、模型初始化、每轮参与者
采样（采样 RNG 的 BitGenerator 状态写入 checkpoint）、每个客户端的
掉线/失败判定、本地训练与聚合结果。相同 seed 的两次运行最终参数逐位一致。

## Checkpoint

每轮结束原子写入（临时文件 + `os.replace`）一个 pickle，包含：
配置、当前轮次、全局参数、采样 RNG 状态、逐轮指标历史。恢复时数据由相同
seed 在内存中确定性重建，接着原轮次继续——从任意轮恢复得到的最终模型与
不中断执行**在浮点容差内（rtol=1e-12 逐轮校验）一致**。

恢复训练允许改变 `--rounds`（训练到更多轮）和 checkpoint 路径；但影响
数据与训练轨迹的关键配置（seed、客户端数、样本数、α、学习率等）改变会
被拒绝并报错。

## 使用方法

无需安装（要求 Python 3.9+、numpy）：

```bash
pip install -r requirements.txt          # numpy + pytest

# IID 训练
python -m fedavg.main --iid --rounds 20 --seed 42

# non-IID + 规模异构 + 随机掉线/失败 + checkpoint
python -m fedavg.main --no-iid --alpha 0.3 --heterogeneous-sizes \
    --drop-prob 0.15 --fail-prob 0.1 \
    --rounds 25 --seed 7 --checkpoint ckpt.pkl

# 从 checkpoint 恢复并训练到更多轮
python -m fedavg.main --resume --checkpoint ckpt.pkl --rounds 40
```

主要参数：`--clients`、`--samples`、`--features`、`--participation`、
`--local-epochs`、`--lr`、`--l2`、`--alpha`。逐轮在终端打印
参与者、成功/掉线/失败客户端、有效样本总数、loss 和 accuracy。

## 测试

```bash
python -m pytest                      # 全部 31 个用例，终端输出 loss/accuracy
python -m pytest -v                  # 详细列表
python -m pytest tests/test_simulator.py::test_checkpoint_resume_matches_uninterrupted
```

覆盖场景：

- **数据**：IID/non-IID 划分正确性、无重叠无遗漏、α 控制偏斜程度、
  规模异构、划分确定性；
- **聚合**：按有效样本数加权（非简单平均）、权重归一化、空更新、
  NaN/Inf 拒绝、本地训练 loss 下降；
- **端到端**：IID（acc>0.80）与 non-IID（acc>0.70）收敛、
  异构规模下每轮 `n_effective` 等于成功客户端真实样本数之和；
- **失败**：随机掉线/失败下训练不中断、部分失败时仅存活者参与聚合、
  全员失败轮跳过且参数不变；
- **确定性**：同 seed 两次运行（含采样与失败序列、最终权重）完全一致，
  不同 seed 产生不同轨迹；
- **checkpoint**：在第 1/5/10/19 轮等多点恢复与不中断执行逐轮一致，
  掉线/失败场景下恢复一致，关键配置变更被拒绝；
- **本地化**：AST 扫描全部源码确认无网络/数据库/第三方服务导入，
  并在把 `socket.socket` / `getaddrinfo` 替换为直接报错的情况下
  完整运行训练（任何网络尝试都会使测试失败）。

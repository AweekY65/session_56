# 本地 FedAvg 联邦学习模拟器

一个**完全本地**的联邦学习（Federated Averaging, FedAvg）模拟器：合成数据由
代码生成，所有客户端数据、模型参数、训练状态、指标只保存在**内存或本地文件**中。
整个项目只有 `numpy` 一个运行时依赖，**不连接**任何模型服务、云训练平台、
数据库或外部服务，也**不下载**任何数据集或预训练模型。测试套件通过
`conftest.py` 直接禁用 socket，从机制上保证运行期间没有任何网络连接。

## 目录结构

```
fedavg_sim/
  data.py         # 合成二分类数据生成 + IID / non-IID 划分
  model.py        # 逻辑回归、BCE 损失、mini-batch SGD 本地训练
  client.py       # 本地 Client：持有数据、执行本地训练、可模拟失败
  aggregation.py  # 按有效样本数加权的 FedAvg
  simulation.py   # 采样 / 掉线 / 失败 / 聚合 / 评估 / checkpoint 主循环
main.py           # 终端 demo，逐轮打印 loss/accuracy
tests/            # 自动化测试（pytest），覆盖全部要求场景
```

## 安装

```bash
pip install -r requirements.txt   # numpy, pytest
```

## FedAvg 流程

每一轮通信（round）执行：

1. **参与者采样**：服务器从可参与的 client 中按固定随机种子无放回采样
   `n_participants` 个（`offline_client_ids` 中的永久离线 client 不参与）。
2. **掉线判定**：每个被采样 client 以 `dropout_prob` 概率本轮掉线，
   掉线者不执行本地训练。
3. **本地训练**：在线 client 收到当前全局模型 `w_t`，在自己的本地数据上跑
   `local_epochs` 个 epoch 的 mini-batch SGD（逻辑回归 + BCE 损失），
   返回更新后的参数与本地训练损失。
4. **失败处理**：训练可能失败（见下节），单个失败 client 的更新被丢弃，
   不影响本轮其他 client。
5. **FedAvg 聚合**：对所有**成功** client 的参数按**有效样本数**加权：

   $$w_{t+1} = \frac{\sum_{k \in S_t} n_k\, w_k^{t+1}}{\sum_{k \in S_t} n_k}$$

   其中 `n_k` 是 client k 本轮实际参与训练的有效样本数（即其本地数据量）。
   **绝不是简单平均**——一个 2000 样本 client 的权重是 10 样本 client 的 200 倍。
6. **评估**：在统一的全局测试集上计算 BCE loss 与 accuracy 并记录。
7. **checkpoint**：按 `checkpoint_every` 将模型、状态与配置原子写入本地目录。

若本轮没有任何成功 client（或成功数低于 `min_successful_participants`），
该轮标记为 `SKIP`、全局模型保持不变，训练继续进行而不是崩溃。

## 数据生成方式

全部数据由 numpy 代码生成，无任何下载：

```
X ~ N(0, I)                         # d 维高斯特征（默认 10 维）
logit = X @ w_true + 0.3 + 噪声      # 线性 logit + 高斯噪声
y ~ Bernoulli(sigmoid(logit))       # 二分类标签 {0, 1}
```

默认 2000 个样本，其中 20% 留出为全局测试集（不划分给任何 client），
其余划分给各 client。支持三种划分/倾斜方式，可叠加：

- **IID**（`--iid`）：按 client 容量对每个类别做分层发牌，各 client 的
  标签比例与总体一致。
- **non-IID 标签倾斜**（默认开启）：每个 client 抽取一个
  `Dirichlet(alpha, alpha)` 标签偏好（`--label-alpha`，默认 0.5；越小越倾斜，
  如 0.05 时部分 client 几乎只有单一标签），在固定 client 容量约束下分配样本，
  保证样本不重不漏全部分完。
- **数量倾斜**（`--sizes-dirichlet`）：各 client 的数据量按
  `Dirichlet(beta,…,beta)` 抽取，`beta` 越小规模越悬殊（如 0.3 时可能出现
  1000+ 与个位数样本并存的 client）。

## 失败语义

| 情况 | 触发方式 | 语义 |
|------|----------|------|
| 永久离线 | `offline_client_ids` | 永远不会被采样 |
| 随机掉线 | `dropout_prob` | 被采样但本轮不训练，不产生更新 |
| 随机训练失败 | `failure_prob` | 被判失败，更新被丢弃 |
| 确定性失败 | `always_fail_client_ids` | 一旦被采样必然训练失败（用于测试） |
| 数据/数值失败 | 空数据、非法标签、参数或结果为 NaN/Inf | 抛 `LocalTrainingError`，该 client 被跳过 |
| 非法聚合输入 | 更新含 NaN/Inf、有效样本数 ≤ 0 | 聚合器拒绝该更新，仅聚合合法更新 |

关键保证：

- **单个失败 client 不会破坏整轮合法聚合**——只聚合成功者；
- 掉线/失败 client 仍按自己的“被采样位置”消耗随机数，因此其他 client 的
  训练种子、以及下一轮行为都与谁掉线/失败无关（失败隔离 + 可复现）；
- 全员失败时该轮 `SKIP`，模型不变，训练继续；
- 若一轮内合法更新总数为零，聚合器抛 `NoValidUpdatesError`，由主循环兜底跳过。

## 确定性 / 可复现

固定 `seed` 后以下全部确定性可复现：数据生成、训练/测试切分、client 划分、
每轮参与者采样、掉线、失败判定、本地 mini-batch 洗牌。实现上所有随机流都从
`SeedSequence(seed)` 按“轮次 → 用途（采样/掉线/失败/训练）→ 采样位置”
派生子流，因此恢复训练**不需要保存 RNG 内部状态**：轮次相同，派生出的随机流
就相同。所有参数计算使用 float64；测试中断言恢复后模型与不中断执行**逐位一致**
（容差 1e-12，实际完全相同）。

## Checkpoint

checkpoint 目录下三个本地文件，均为“写临时文件 + 原子 rename”：

- `config.json`：完整训练配置（恢复时校验，不一致直接报错）；
- `state.json`：下一轮编号、每轮指标、数据指纹（client 数/各 client 样本数/
  特征维度/测试集大小，恢复时校验必须与传入数据一致）；
- `model.npz`：全局模型 `w, b`（float64）。

恢复时需用**相同 seed/参数重新生成**同一份合成数据（数据本身不入 checkpoint）。

## 使用示例

```bash
# 基本 non-IID 训练（逐轮打印采样/掉线/失败/有效样本数/train_loss/test_loss/acc）
python3 main.py

# IID
python3 main.py --iid --rounds 30 --seed 1

# 8 client、每轮选 5 个；25% 掉线、15% 训练失败；标签与数量双重 non-IID；带 checkpoint
python3 main.py --rounds 14 --n-clients 8 --participants 5 \
    --dropout-prob 0.25 --failure-prob 0.15 \
    --label-alpha 0.3 --sizes-dirichlet 0.3 \
    --checkpoint-dir ./ckpt

# 指定某些 client 永久离线 / 必失败
python3 main.py --offline-ids 0 --always-fail-ids 3 7

# 从 checkpoint 恢复并继续（配置从 checkpoint 读取，其余数据参数仍需一致）
python3 main.py --resume ./ckpt --n-clients 8 --participants 5 \
    --dropout-prob 0.25 --failure-prob 0.15 \
    --sizes-dirichlet 0.3 --label-alpha 0.3 --seed 42
```

终端输出示例：

```
round 07 [ok  ] 采样=[7,0,3,1,5] 掉线=[3] 失败=[1] 贡献=[0,5,7] 有效样本=400 \
    train_loss=0.2481 test_loss=0.3423 acc=0.8625
round 10 [ok  ] ... train_loss=0.2292 test_loss=0.3332 acc=0.8600
```

## 运行测试

```bash
python3 -m pytest tests/ -v          # 29 个测试
python3 -m pytest tests/ -v -s       # 同时打印每轮 loss/accuracy
```

测试覆盖：

| 测试 | 覆盖场景 |
|------|----------|
| `test_iid_training_improves_loss_and_accuracy` | IID 下 loss 下降、accuracy 提升 |
| `test_noniid_training_still_learns` | non-IID（Dirichlet 倾斜）下仍能学习 |
| `test_iid_partition_label_ratios...` / `test_non_iid_label_skew...` / `test_size_skew...` | IID/non-IID 划分正确性、标签与数量倾斜 |
| `test_unequal_client_sizes_weighted_correctly` | 规模悬殊 client：手工复现聚合，验证按 n 加权而非简单平均 |
| `test_weighted_by_effective_samples_not_simple_mean` 等 | FedAvg 加权公式、顺序无关、权重和为 1 |
| `test_dropouts_do_not_break_round` / `test_permanent_offline...` | 随机掉线、永久离线 |
| `test_failing_clients_are_skipped...` | 随机+确定性失败下整轮聚合仍合法、模型仍收敛 |
| `test_all_participants_fail...` / `test_min_successful_participants...` | 全员失败跳过、成功数下限 |
| `test_single_failure_does_not_change_other...` | 失败 client 不改变其他 client 的本地更新 |
| `test_rejects_non_finite...` / `test_all_invalid_raises` | NaN/Inf 更新拒绝、空聚合语义 |
| `test_checkpoint_resume_matches_uninterrupted` | 中断恢复后最终模型/每轮指标与不中断一致（1e-12） |
| `test_checkpoint_resume_at_every_round` | 任意轮次边界（1..7）恢复都一致 |
| `test_determinism_same_seed_identical_runs` | 同 seed 两次运行逐位一致（采样/掉线/失败/参数） |
| `test_different_seeds_change_sampling` | 不同 seed 产生不同采样 |
| `tests/conftest.py` | 禁用全部 socket，任何网络尝试都会让测试失败 |

## 设计约束

- 运行时依赖仅 `numpy`；测试依赖 `pytest`。
- 无网络、无多进程/多机通信、无外部存储——“client”“服务器”均为进程内对象，
  便于精确复现与断言。

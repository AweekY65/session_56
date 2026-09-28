"""FedAvg 聚合：按各 client 的有效样本数加权平均参数。

    w_{t+1} = sum_{k∈S_t} n_k * w_k^{t+1} / sum_{k∈S_t} n_k

其中 n_k 是 client k 本轮的**有效样本数**，S_t 是本轮成功提交更新的
client 集合。绝不能对规模不同的 client 做简单平均。

鲁棒性：
- 含 NaN/Inf 的更新会被拒绝（等价于该 client 本轮失败）；
- effective_n_samples 非法（<=0）的更新会被拒绝；
- 所有更新都无效时抛 NoValidUpdatesError，由模拟器决定跳过本轮。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .client import ClientUpdate
from .model import Params


class NoValidUpdatesError(RuntimeError):
    """没有任何合法更新可供聚合。"""


@dataclass(frozen=True)
class AggregatedModel:
    params: Params
    total_effective_samples: int
    contributing_client_ids: tuple[int, ...]
    rejected_client_ids: tuple[int, ...]


def fed_avg(updates: list[ClientUpdate]) -> AggregatedModel:
    """按有效样本数加权聚合多个 client 更新。

    空列表或全部非法时抛 NoValidUpdatesError。
    """
    if not updates:
        raise NoValidUpdatesError("没有收到任何 client 更新")

    accepted: list[ClientUpdate] = []
    rejected: list[int] = []
    for upd in updates:
        if upd.effective_n_samples <= 0:
            rejected.append(upd.client_id)
            continue
        if not np.all(np.isfinite(upd.params.w)) or not np.isfinite(upd.params.b):
            rejected.append(upd.client_id)
            continue
        accepted.append(upd)

    if not accepted:
        raise NoValidUpdatesError("所有 client 更新均无效（样本数非法或参数含 NaN/Inf）")

    # 按 client_id 排序，保证同样的输入聚合结果逐位确定。
    accepted.sort(key=lambda u: u.client_id)
    total_n = sum(u.effective_n_samples for u in accepted)

    w_acc = np.zeros_like(accepted[0].params.w, dtype=np.float64)
    b_acc = 0.0
    for upd in accepted:
        weight = upd.effective_n_samples / total_n
        w_acc += weight * upd.params.w
        b_acc += weight * upd.params.b

    return AggregatedModel(
        params=Params(w=w_acc, b=float(b_acc)),
        total_effective_samples=total_n,
        contributing_client_ids=tuple(u.client_id for u in accepted),
        rejected_client_ids=tuple(rejected),
    )

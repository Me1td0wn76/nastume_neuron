"""最適化手法（勾配を使ってパラメータを更新する方法）。"""

import math

import numpy as np

from .layers import Parameter


class AdamW:
    """Adam + 重み減衰（Weight Decay）。

    Adam は「勾配の移動平均(m)」と「勾配の2乗の移動平均(v)」を使い、
    パラメータごとに歩幅を自動調整する。いまの深層学習でいちばん標準的な手法。
    """

    def __init__(self, params: dict[str, Parameter], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = params
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def step(self, lr: float):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # 学習初期は m, v が 0 に偏っているので補正する
        bias1 = 1 - b1**self.t
        bias2 = 1 - b2**self.t
        for k, p in self.params.items():
            g, m, v = p.grad, self.m[k], self.v[k]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            # 重み減衰は行列（2次元以上）だけにかける。バイアスや LayerNorm にはかけない
            if p.data.ndim >= 2 and self.weight_decay:
                p.data -= lr * self.weight_decay * p.data
            p.data -= lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {"t": np.array(self.t)}
        state.update({f"m/{k}": a for k, a in self.m.items()})
        state.update({f"v/{k}": a for k, a in self.v.items()})
        return state

    def load_state_dict(self, state):
        self.t = int(state["t"])
        for k in self.params:
            self.m[k][...] = state[f"m/{k}"]
            self.v[k][...] = state[f"v/{k}"]


def clip_grad_norm(params: dict[str, Parameter], max_norm: float) -> float:
    """勾配全体の大きさ(ノルム)が max_norm を超えたら縮める。学習の暴走を防ぐ。"""
    total = math.sqrt(sum(float((p.grad * p.grad).sum()) for p in params.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params.values():
            p.grad *= scale
    return total


def cosine_lr(it: int, max_lr: float, warmup: int, max_iters: int, min_lr_ratio: float = 0.1) -> float:
    """学習率のスケジュール: 最初は徐々に上げ（ウォームアップ）、その後コサイン曲線で下げていく。"""
    min_lr = max_lr * min_lr_ratio
    if it < warmup:
        return max_lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    progress = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

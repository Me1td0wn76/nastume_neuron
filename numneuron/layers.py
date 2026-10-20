"""Transformer を構成する層（レイヤー）を NumPy だけで実装する。

各層は forward（順伝播）と backward（逆伝播）を持つ。
- forward:  入力から出力を計算し、逆伝播で使う値を self に保存しておく
- backward: 「出力に対する損失の勾配」を受け取り、「入力に対する勾配」を返す。
            同時に、自分のパラメータの勾配を Parameter.grad に足し込む。

PyTorch などの自動微分（autograd）が裏でやっていることを、手で書いている。

np.tanh などの関数は、xp = array_module(x) で選んだ xp.tanh として呼ぶ。
x が GPU 上の配列なら xp は CuPy になり、計算が GPU で行われる（backend.py を参照）。
"""

import math

import numpy as np

from .backend import array_module, to_device


class Parameter:
    """学習される重み。値(data)と勾配(grad)をセットで持つ。"""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = array_module(data).zeros_like(data)


class Module:
    """層の基底クラス。持っている Parameter を名前付きで集められる。"""

    def parameters(self) -> dict[str, Parameter]:
        params = {}
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                params[name] = value
            elif isinstance(value, Module):
                for sub, p in value.parameters().items():
                    params[f"{name}.{sub}"] = p
            elif isinstance(value, list):
                for i, m in enumerate(value):
                    if isinstance(m, Module):
                        for sub, p in m.parameters().items():
                            params[f"{name}.{i}.{sub}"] = p
        return params

    def zero_grad(self):
        for p in self.parameters().values():
            p.grad[...] = 0

    def to(self, device: str):
        """全パラメータを CPU（"cpu"）か GPU（"gpu"）に移す。自分自身を返す。"""
        for p in self.parameters().values():
            p.data = to_device(p.data, device)
            p.grad = to_device(p.grad, device)
        return self


# ---------------------------------------------------------------------------
# 基本の層
# ---------------------------------------------------------------------------


class Linear(Module):
    """全結合層: y = x @ W + b"""

    def __init__(self, n_in, n_out, rng, std=0.02, dtype=np.float32):
        self.W = Parameter((rng.standard_normal((n_in, n_out)) * std).astype(dtype))
        self.b = Parameter(np.zeros(n_out, dtype=dtype))

    def forward(self, x):
        self.x = x
        # (B, T, n_in) を (B*T, n_in) の2次元にしてから掛けたほうが速い
        y = x.reshape(-1, x.shape[-1]) @ self.W.data + self.b.data
        return y.reshape(*x.shape[:-1], -1)

    def backward(self, dout):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = dout.reshape(-1, dout.shape[-1])
        self.W.grad += x2.T @ d2
        self.b.grad += d2.sum(axis=0)
        dx = d2 @ self.W.data.T
        return dx.reshape(self.x.shape)


class Embedding(Module):
    """埋め込み層: 番号（文字ID など）をベクトルに変換する表引き。"""

    def __init__(self, n_vocab, dim, rng, std=0.02, dtype=np.float32):
        self.W = Parameter((rng.standard_normal((n_vocab, dim)) * std).astype(dtype))

    def forward(self, idx):
        self.idx = idx
        return self.W.data[idx]

    def backward(self, dout):
        # 同じ文字が何度も出てくるので、勾配はその行に「足し込む」
        array_module(dout).add.at(self.W.grad, self.idx.ravel(), dout.reshape(-1, dout.shape[-1]))


class LayerNorm(Module):
    """層正規化: ベクトルごとに平均0・分散1に揃えてから、スケール(gamma)とシフト(beta)を掛ける。"""

    def __init__(self, dim, eps=1e-5, dtype=np.float32):
        self.gamma = Parameter(np.ones(dim, dtype=dtype))
        self.beta = Parameter(np.zeros(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        xp = array_module(x)
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        rstd = 1.0 / xp.sqrt(var + self.eps)
        xhat = (x - mu) * rstd
        self.xhat, self.rstd = xhat, rstd
        return xhat * self.gamma.data + self.beta.data

    def backward(self, dout):
        xhat, rstd = self.xhat, self.rstd
        C = xhat.shape[-1]
        self.gamma.grad += (dout * xhat).reshape(-1, C).sum(axis=0)
        self.beta.grad += dout.reshape(-1, C).sum(axis=0)
        dxhat = dout * self.gamma.data
        # 平均と分散も x に依存しているので、その分の勾配を差し引く
        return rstd * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
        )


_GELU_K = math.sqrt(2.0 / math.pi)


class GELU(Module):
    """活性化関数 GELU（GPT で使われる tanh 近似版）。ReLU を滑らかにしたような形。"""

    def forward(self, x):
        t = array_module(x).tanh(_GELU_K * (x + 0.044715 * x * x * x))
        self.x, self.t = x, t
        return 0.5 * x * (1.0 + t)

    def backward(self, dout):
        x, t = self.x, self.t
        dt = _GELU_K * (1.0 + 3 * 0.044715 * x * x)
        return dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dt)


def softmax(x, axis=-1):
    """最大値を引いてから exp を取ると、オーバーフローしない。"""
    e = array_module(x).exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Transformer の部品
# ---------------------------------------------------------------------------


class CausalSelfAttention(Module):
    """因果的（未来を見ない）マルチヘッド自己注意機構。

    各文字が「それより前の文字のうち、どれに注目するか」を学習する。
    """

    def __init__(self, n_embd, n_head, n_layer, rng, dtype=np.float32):
        assert n_embd % n_head == 0, "n_embd は n_head で割り切れる必要があります"
        self.n_head = n_head
        self.qkv = Linear(n_embd, 3 * n_embd, rng, dtype=dtype)
        # 残差接続に足し込む層は、層の数に応じて初期値を小さくする（GPT-2 の工夫）
        self.proj = Linear(n_embd, n_embd, rng, std=0.02 / math.sqrt(2 * n_layer), dtype=dtype)

    def forward(self, x):
        xp = array_module(x)
        B, T, C = x.shape
        H = self.n_head
        D = C // H

        # Query（何を探すか）, Key（自分は何か）, Value（中身）を一度に作り、ヘッドごとに分ける
        qkv = self.qkv.forward(x).reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)
        q, k, v = (xp.ascontiguousarray(a) for a in qkv)  # それぞれ (B, H, T, D)

        scale = 1.0 / math.sqrt(D)
        score = (q @ k.swapaxes(-1, -2)) * scale  # (B, H, T, T): 各文字ペアの関連度
        mask = xp.tril(xp.ones((T, T), dtype=bool))  # 自分より後ろ（未来）の文字は見せない
        score = xp.where(mask, score, -xp.inf)
        att = softmax(score)  # 各行が「注目の割合」になる（合計 1）
        y = att @ v  # 注目した文字の Value を混ぜ合わせる

        self.q, self.k, self.v, self.att, self.scale = q, k, v, att, scale
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # ヘッドを結合
        return self.proj.forward(y)

    def backward(self, dout):
        q, k, v, att, scale = self.q, self.k, self.v, self.att, self.scale
        B, H, T, D = q.shape

        dy = self.proj.backward(dout).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        datt = dy @ v.swapaxes(-1, -2)
        dv = att.swapaxes(-1, -2) @ dy
        # softmax の逆伝播（マスクされた位置は att=0 なので勾配も自動的に 0）
        dscore = att * (datt - (datt * att).sum(axis=-1, keepdims=True)) * scale
        dq = dscore @ k
        dk = dscore.swapaxes(-1, -2) @ q

        # forward で行った reshape / transpose を逆向きにたどる
        dqkv = array_module(dq).stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * H * D)
        return self.qkv.backward(dqkv)


class MLP(Module):
    """各文字のベクトルを個別に変換する2層のニューラルネット（中間は4倍に広げる）。"""

    def __init__(self, n_embd, n_layer, rng, dtype=np.float32):
        self.fc = Linear(n_embd, 4 * n_embd, rng, dtype=dtype)
        self.act = GELU()
        self.proj = Linear(4 * n_embd, n_embd, rng, std=0.02 / math.sqrt(2 * n_layer), dtype=dtype)

    def forward(self, x):
        return self.proj.forward(self.act.forward(self.fc.forward(x)))

    def backward(self, dout):
        return self.fc.backward(self.act.backward(self.proj.backward(dout)))


class Block(Module):
    """Transformer ブロック = 注意機構 + MLP（それぞれ LayerNorm と残差接続つき）。"""

    def __init__(self, n_embd, n_head, n_layer, rng, dtype=np.float32):
        self.ln1 = LayerNorm(n_embd, dtype=dtype)
        self.attn = CausalSelfAttention(n_embd, n_head, n_layer, rng, dtype=dtype)
        self.ln2 = LayerNorm(n_embd, dtype=dtype)
        self.mlp = MLP(n_embd, n_layer, rng, dtype=dtype)

    def forward(self, x):
        h = x + self.attn.forward(self.ln1.forward(x))
        return h + self.mlp.forward(self.ln2.forward(h))

    def backward(self, dout):
        # 残差接続 (h = x + f(x)) の勾配は「そのまま流れる分」+「f を通る分」
        dh = dout + self.ln2.backward(self.mlp.backward(dout))
        return dh + self.ln1.backward(self.attn.backward(dh))


# ---------------------------------------------------------------------------
# 損失関数
# ---------------------------------------------------------------------------


def cross_entropy(logits, targets):
    """交差エントロピー損失（次の文字の予測がどれだけ外れたか）と、logits に対する勾配を返す。

    logits: (B, T, V) 各位置で「次に来る文字」のスコア
    targets: (B, T) 正解の次の文字ID（CPU 上の配列でもよい。logits と同じ側に移して使う）
    """
    xp = array_module(logits)
    B, T, V = logits.shape
    x = logits.reshape(-1, V)
    t = xp.asarray(targets).ravel()
    n = t.size
    rows = xp.arange(n)

    # log softmax を安定に計算: log p = x - logsumexp(x)
    m = x.max(axis=1, keepdims=True)
    lse = m + xp.log(xp.exp(x - m).sum(axis=1, keepdims=True))
    logp = x - lse
    loss = -logp[rows, t].mean()

    # softmax + 交差エントロピーの勾配は (予測確率 - 正解の one-hot) / n というきれいな形になる
    dx = xp.exp(logp)
    dx[rows, t] -= 1.0
    dx /= n
    return float(loss), dx.reshape(B, T, V)

"""文字単位の小さな GPT（デコーダのみの Transformer）。

流れ:
    文字ID列 ─→ [文字の埋め込み + 位置の埋め込み + 文体(作家)の埋め込み]
             ─→ Transformer ブロック × n_layer
             ─→ LayerNorm ─→ 各位置で「次の文字」のスコア(logits)
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .backend import array_module, to_numpy
from .layers import Block, Embedding, LayerNorm, Module, softmax


@dataclass
class GPTConfig:
    vocab_size: int  # 文字の種類数
    n_styles: int = 1  # 文体（作家）の数
    block_size: int = 128  # 一度に読める文字数（文脈長）
    n_layer: int = 4  # Transformer ブロックの数
    n_head: int = 4  # 注意機構のヘッド数
    n_embd: int = 192  # 文字ベクトルの次元数


class GPT(Module):
    def __init__(self, config: GPTConfig, seed: int = 0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        c = config
        self.config = c
        self.tok_emb = Embedding(c.vocab_size, c.n_embd, rng, dtype=dtype)
        self.pos_emb = Embedding(c.block_size, c.n_embd, rng, dtype=dtype)
        # 作家ごとのベクトル。全位置に足すことで「誰っぽく書くか」を指定できる
        self.style_emb = Embedding(c.n_styles, c.n_embd, rng, dtype=dtype)
        self.blocks = [Block(c.n_embd, c.n_head, c.n_layer, rng, dtype=dtype) for _ in range(c.n_layer)]
        self.ln_f = LayerNorm(c.n_embd, dtype=dtype)

    def forward(self, idx, style, last_only=False):
        """idx: (B, T) 文字ID, style: (B,) 作家ID → logits: (B, T, V)

        last_only=True なら最後の位置の logits だけを (B, 1, V) で返す（生成時の高速化）。
        idx と style は CPU 上の配列でもよい。重みが GPU にあれば、GPU に送ってから計算する。
        """
        xp = array_module(self.tok_emb.W.data)
        idx, style = xp.asarray(idx), xp.asarray(style)
        B, T = idx.shape
        assert T <= self.config.block_size, f"入力が長すぎます（{T} > {self.config.block_size}）"
        x = (
            self.tok_emb.forward(idx)
            + self.pos_emb.forward(xp.arange(T))[None, :, :]
            + self.style_emb.forward(style)[:, None, :]
        )
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        if last_only:
            x = x[:, -1:, :]
        self.x_final = x

        # 出力層は文字の埋め込み行列を使い回す（重み共有: パラメータ数を減らせる）
        W = self.tok_emb.W.data
        return (x.reshape(-1, x.shape[-1]) @ W.T).reshape(B, x.shape[1], -1)

    def backward(self, dlogits):
        x = self.x_final
        B, T, C = x.shape
        V = dlogits.shape[-1]
        W = self.tok_emb.W
        d2 = dlogits.reshape(-1, V)

        # logits = x @ W^T の逆伝播（W は入力の埋め込みと共有しているので勾配が合算される）
        W.grad += d2.T @ x.reshape(-1, C)
        dx = (d2 @ W.data).reshape(B, T, C)

        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # 3つの埋め込みは足し算で合流していたので、同じ勾配がそれぞれに流れる
        self.tok_emb.backward(dx)
        self.pos_emb.backward(dx.sum(axis=0))  # 位置の埋め込みはバッチ全体で共通
        self.style_emb.backward(dx.sum(axis=1))  # 文体の埋め込みは全位置で共通

    def generate(
        self,
        idx: list[int],
        style: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        banned: list[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Iterator[int]:
        """idx の続きを1文字ずつ生成して返し続けるジェネレータ。

        temperature: 小さいほど無難（確率の高い文字ばかり）、大きいほど冒険的になる
        top_k:       確率上位 k 個の文字からだけ選ぶ（変な文字が出にくくなる）
        banned:      絶対に出さない文字ID（<unk> など）
        """
        rng = rng or np.random.default_rng()
        idx = list(idx)
        style_arr = np.array([style])
        while True:
            context = np.array([idx[-self.config.block_size :]])
            # 文字を選ぶ乱数は CPU（NumPy）で扱うので、スコアは CPU に持ってくる
            logits = to_numpy(self.forward(context, style_arr, last_only=True)[0, -1]).astype(np.float64)
            if banned:
                logits[banned] = -np.inf
            logits /= max(temperature, 1e-6)
            if top_k is not None and top_k < len(logits):
                threshold = np.partition(logits, -top_k)[-top_k]
                logits[logits < threshold] = -np.inf
            probs = softmax(logits)
            next_id = int(rng.choice(len(probs), p=probs))
            idx.append(next_id)
            yield next_id

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters().values())

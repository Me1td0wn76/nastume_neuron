"""GPU（CuPy）で計算しても、CPU（NumPy）と同じ結果になるかを確かめる。

CuPy が入っていない、または GPU がない環境ではスキップする。
"""

import numpy as np
import pytest

from numneuron.backend import gpu_available, to_numpy
from numneuron.checkpoint import load_weights, save_weights
from numneuron.layers import cross_entropy
from numneuron.model import GPT, GPTConfig
from numneuron.optim import AdamW, clip_grad_norm

pytestmark = pytest.mark.skipif(not gpu_available(), reason="GPU が使えない")

CONFIG = GPTConfig(vocab_size=50, n_styles=2, block_size=16, n_layer=2, n_head=4, n_embd=32)


def make_batch():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, CONFIG.vocab_size, size=(4, CONFIG.block_size))
    targets = rng.integers(0, CONFIG.vocab_size, size=(4, CONFIG.block_size))
    return idx, np.array([0, 1, 0, 1]), targets


def test_gpu_matches_cpu():
    idx, style, targets = make_batch()
    results = {}
    for device in ("cpu", "gpu"):
        model = GPT(CONFIG, seed=0).to(device)  # 同じ seed なら、同じ初期値から始まる
        loss, dlogits = cross_entropy(model.forward(idx, style), targets)
        model.zero_grad()
        model.backward(dlogits)
        grads = {k: to_numpy(p.grad).copy() for k, p in model.parameters().items()}
        results[device] = loss, grads

    (cpu_loss, cpu_grads), (gpu_loss, gpu_grads) = results["cpu"], results["gpu"]
    assert np.isclose(cpu_loss, gpu_loss, rtol=1e-5)
    for k in cpu_grads:
        np.testing.assert_allclose(gpu_grads[k], cpu_grads[k], rtol=1e-3, atol=1e-6, err_msg=k)


def test_train_on_gpu_then_load_on_cpu(tmp_path):
    idx, style, targets = make_batch()
    model = GPT(CONFIG, seed=0).to("gpu")
    params = model.parameters()
    optimizer = AdamW(params)
    losses = []
    for _ in range(20):  # 同じバッチを何度も見せれば、損失は下がるはず
        loss, dlogits = cross_entropy(model.forward(idx, style), targets)
        model.zero_grad()
        model.backward(dlogits)
        clip_grad_norm(params, 1.0)
        optimizer.step(1e-2)
        losses.append(loss)
    assert losses[-1] < losses[0]

    # GPU の重みを保存し、CPU のモデルに読み込むと、同じ出力になる
    save_weights(tmp_path / "model.npz", model, **optimizer.state_dict())
    cpu_model = GPT(CONFIG, seed=1)
    state = load_weights(tmp_path / "model.npz", cpu_model)
    np.testing.assert_allclose(
        cpu_model.forward(idx, style), to_numpy(model.forward(idx, style)), rtol=1e-4, atol=1e-5
    )

    # 最適化手法の状態は GPU に読み戻せる（学習の再開）
    resumed = AdamW(model.parameters())
    resumed.load_state_dict(state)
    assert resumed.t == optimizer.t
    k = next(iter(params))
    np.testing.assert_array_equal(to_numpy(resumed.m[k]), to_numpy(optimizer.m[k]))


def test_generate_on_gpu():
    model = GPT(CONFIG, seed=0).to("gpu")
    gen = model.generate([1, 2, 3], style=1, rng=np.random.default_rng(0))
    ids = [next(gen) for _ in range(30)]  # 文脈長(16)を超えても生成し続けられる
    assert all(0 <= i < CONFIG.vocab_size for i in ids)

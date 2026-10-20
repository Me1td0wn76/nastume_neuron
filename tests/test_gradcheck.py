"""手書きの逆伝播が正しいかを「数値微分」で確かめる（勾配確認）。

パラメータを ±eps だけずらしたときの損失の変化 (L(w+eps) - L(w-eps)) / 2eps は
勾配の近似値になる。これと backward で計算した勾配が一致すれば、逆伝播は正しい。
"""

import numpy as np
import pytest

from numneuron.backend import gpu_available, to_numpy
from numneuron.layers import cross_entropy
from numneuron.model import GPT, GPTConfig

# GPU（CuPy）が使える環境では、GPU でも同じ確認をする
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not gpu_available(), reason="GPU が使えない"))]


def compute_loss(model, idx, style, targets, backward=False):
    model.zero_grad()
    logits = model.forward(idx, style)
    loss, dlogits = cross_entropy(logits, targets)
    if backward:
        model.backward(dlogits)
    return loss


@pytest.mark.parametrize("device", DEVICES)
def test_gpt_gradients_match_numerical(device):
    config = GPTConfig(vocab_size=13, n_styles=2, block_size=8, n_layer=2, n_head=2, n_embd=8)
    # 誤差を見分けやすいよう、勾配確認だけは倍精度(float64)で行う
    model = GPT(config, seed=0, dtype=np.float64)
    rng = np.random.default_rng(1)
    # 初期値のままだと値が小さく差が出にくいので、重みを少し揺らしておく
    for p in model.parameters().values():
        p.data += rng.standard_normal(p.data.shape) * 0.3
    model.to(device)

    idx = rng.integers(0, config.vocab_size, size=(3, 6))  # 文脈長より短い入力でも確認
    targets = rng.integers(0, config.vocab_size, size=(3, 6))
    style = np.array([0, 1, 1])

    compute_loss(model, idx, style, targets, backward=True)
    params = model.parameters()
    analytic = {k: to_numpy(p.grad).copy() for k, p in params.items()}

    eps = 1e-6
    for name, p in params.items():
        flat = p.data.reshape(-1)
        picks = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        numerical = []
        for i in picks:
            original = float(flat[i])
            flat[i] = original + eps
            plus = compute_loss(model, idx, style, targets)
            flat[i] = original - eps
            minus = compute_loss(model, idx, style, targets)
            flat[i] = original
            numerical.append((plus - minus) / (2 * eps))
        np.testing.assert_allclose(
            analytic[name].reshape(-1)[picks], numerical, rtol=1e-4, atol=1e-8, err_msg=name
        )


def test_cross_entropy_of_uniform_prediction():
    # すべての文字が同じスコアなら、損失は log(語彙数) になる
    V = 50
    loss, _ = cross_entropy(np.zeros((2, 3, V)), np.zeros((2, 3), dtype=int))
    assert np.isclose(loss, np.log(V))


def test_attention_does_not_look_ahead():
    # 未来の文字を変えても、それより前の位置の出力は変わらないはず
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(config, seed=0)
    idx = np.array([[1, 2, 3, 4, 5, 6]])
    changed = np.array([[1, 2, 3, 9, 9, 9]])
    style = np.array([0])
    a = model.forward(idx, style)
    b = model.forward(changed, style)
    np.testing.assert_allclose(a[:, :3], b[:, :3], rtol=1e-5)
    assert not np.allclose(a[:, 3:], b[:, 3:])

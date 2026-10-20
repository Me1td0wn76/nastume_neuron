"""計算を CPU（NumPy）でするか、GPU（CuPy）でするかの切り替え。

CuPy は、NumPy とほぼ同じ書き方で NVIDIA の GPU を使って計算できるライブラリ。
np.tanh(x) の代わりに xp.tanh(x) と書き、xp に numpy か cupy を入れておけば、
同じコードが CPU でも GPU でも動く。各層では、入力の配列がどちらに載っているかを
array_module(x) で調べて xp を決めている。

GPU を使うには CuPy を入れる:  uv sync --extra gpu
"""

import numpy as np

try:
    import cupy as cp
except ImportError:  # CuPy が入っていなければ CPU だけで動かす
    cp = None

DEVICES = ("auto", "cpu", "gpu")


def gpu_available() -> bool:
    """CuPy が入っていて、使える NVIDIA の GPU があるか。"""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:  # ドライバが入っていない、など
        return False


def resolve_device(device: str) -> str:
    """"auto" なら、GPU が使えるときは "gpu"、使えなければ "cpu" にする。"""
    if device == "auto":
        return "gpu" if gpu_available() else "cpu"
    if device == "gpu" and not gpu_available():
        if cp is None:
            raise SystemExit("GPU を使うには CuPy が必要です。`uv sync --extra gpu` でインストールしてください")
        raise SystemExit("NVIDIA の GPU が見つかりません（ドライバが入っているか確認してください）")
    return device


def device_name(device: str) -> str:
    """表示用の名前。"""
    if device == "gpu":
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return f"GPU（{props['name'].decode()}）"
    return "CPU"


def array_module(x):
    """配列 x が載っている側のライブラリ（numpy か cupy）を返す。"""
    if cp is None:
        return np
    return cp.get_array_module(x)


def to_numpy(x) -> np.ndarray:
    """GPU 上の配列を CPU（NumPy）に持ってくる。保存や乱数による文字選びの前に使う。"""
    if cp is not None and isinstance(x, cp.ndarray):
        return x.get()
    return np.asarray(x)


def to_device(x, device: str):
    """配列を CPU（"cpu"）か GPU（"gpu"）に移す。"""
    return cp.asarray(x) if device == "gpu" else to_numpy(x)

"""学習済みモデルの保存と読み込み。

checkpoints/<名前>/
    config.json  モデルの設定・文字の辞書・作家の一覧
    model.npz    検証データで一番成績が良かったときの重み（生成に使う）
    state.npz    最新の重みと最適化手法の状態（学習の再開に使う）
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backend import array_module, to_numpy
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def save_config(ckpt_dir: Path, config: GPTConfig, tokenizer: CharTokenizer, authors: list[str], **extra):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": asdict(config), "authors": authors, "itos": tokenizer.itos, **extra}
    (ckpt_dir / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def save_weights(path: Path, model: GPT, **extra_arrays):
    arrays = {f"param/{k}": p.data for k, p in model.parameters().items()}
    arrays.update(extra_arrays)
    arrays = {k: to_numpy(a) for k, a in arrays.items()}  # GPU 上の配列は CPU に移してから保存する
    tmp = path.with_name(path.stem + ".tmp.npz")  # 書き込み途中で止まっても壊れないように
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_weights(path: Path, model: GPT) -> dict[str, np.ndarray]:
    """重みをモデルに読み込み、ファイル内の全配列を返す（最適化手法の状態などを取り出すため）。"""
    data = dict(np.load(path))
    for k, p in model.parameters().items():
        p.data[...] = array_module(p.data).asarray(data[f"param/{k}"])
    return data


def load_model(ckpt_dir: Path, device: str = "cpu") -> tuple[GPT, CharTokenizer, list[str]]:
    meta = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    model = GPT(GPTConfig(**meta["model"]))
    load_weights(ckpt_dir / "model.npz", model)
    return model.to(device).eval(), CharTokenizer(meta["itos"]), meta["authors"]

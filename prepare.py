"""ステップ2: ダウンロードしたテキストを整形し、学習用の数値データに変換する。

使い方:
    uv run python prepare.py                     # data/raw/ にある全作家を使う
    uv run python prepare.py --authors 夏目漱石   # 作家を選ぶ

出力: data/processed/
    dataset.npz         文字ID列（学習用・検証用）と、各文字がどの作家のものか
    meta.json           文字の辞書と作家の一覧
    corpus_<作家>.txt    整形後の本文（中身を目で確認する用）
"""

import argparse
import json
from pathlib import Path

import numpy as np

from numneuron.aozora import clean_aozora
from numneuron.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="青空文庫テキストを学習用データに変換する")
    parser.add_argument("--authors", nargs="*", help="使う作家（省略すると data/raw/ にある全員）")
    parser.add_argument("--min-count", type=int, default=3, help="これ未満しか出てこない文字は <unk> にまとめる")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="各作品の末尾から検証用に取っておく割合")
    args = parser.parse_args()

    authors = args.authors or sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    if not authors:
        raise SystemExit(f"{RAW_DIR} にデータがありません。先に download.py を実行してください")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = []  # (作家ID, 本文)
    for style_id, author in enumerate(authors):
        files = sorted((RAW_DIR / author).glob("*.txt"))
        texts = [t for t in (clean_aozora(f.read_text(encoding="utf-8")) for f in files) if t]
        docs.extend((style_id, t) for t in texts)
        (OUT_DIR / f"corpus_{author}.txt").write_text("\n\n".join(texts), encoding="utf-8")
        print(f"{author}: {len(texts)} 作品, {sum(map(len, texts)):,} 文字")

    tokenizer = CharTokenizer.build((t for _, t in docs), min_count=args.min_count)

    # 各作品の前 95% を学習用、後ろ 5% を検証用（学習に使わず、未知の文章への性能を測る）にする
    splits = {"train": ([], []), "val": ([], [])}
    for style_id, text in docs:
        ids = np.array(tokenizer.encode(text + "\n\n"), dtype=np.uint16)  # 作品の区切りは空行
        cut = len(ids) - int(len(ids) * args.val_ratio)
        for name, part in (("train", ids[:cut]), ("val", ids[cut:])):
            splits[name][0].append(part)
            splits[name][1].append(np.full(len(part), style_id, dtype=np.uint8))

    arrays = {}
    for name, (tokens, styles) in splits.items():
        arrays[f"{name}_tokens"] = np.concatenate(tokens)
        arrays[f"{name}_styles"] = np.concatenate(styles)
    np.savez(OUT_DIR / "dataset.npz", **arrays)

    meta = {"authors": authors, "itos": tokenizer.itos}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    all_tokens = np.concatenate([arrays["train_tokens"], arrays["val_tokens"]])
    unk_rate = (all_tokens == tokenizer.unk_id).mean()
    print(f"文字の種類（語彙数）: {len(tokenizer):,}（<unk> になった文字の割合: {unk_rate:.3%}）")
    print(f"学習用: {len(arrays['train_tokens']):,} 文字 / 検証用: {len(arrays['val_tokens']):,} 文字")
    print(f"保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()

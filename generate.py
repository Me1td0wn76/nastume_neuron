"""ステップ4: 学習したモデルで文章を生成する（text to text）。

入力した文章を「書き出し」として読み込み、その続きを文豪風に書く。

使い方:
    uv run python generate.py                               # 対話モード
    uv run python generate.py --prompt "吾輩は猫である。"     # 1回だけ生成
    uv run python generate.py --ckpt checkpoints/tiny        # 使うモデルを選ぶ
"""

import argparse
from pathlib import Path

import numpy as np

from numneuron.checkpoint import load_model

ROOT = Path(__file__).resolve().parent
STOP_CHARS = "。」！？\n"  # 文の切れ目


def find_latest_checkpoint() -> Path:
    candidates = [d for d in (ROOT / "checkpoints").glob("*") if (d / "model.npz").exists()]
    if not candidates:
        raise SystemExit("学習済みモデルがありません。先に train.py を実行してください")
    return max(candidates, key=lambda d: (d / "model.npz").stat().st_mtime)


def write_continuation(model, tokenizer, prompt, style, length, temperature, top_k, rng) -> str:
    """prompt の続きを1文字ずつ表示しながら生成する。"""
    unknown = tokenizer.unknown_chars(prompt)
    if unknown:
        print(f"（辞書にない文字は {tokenizer.UNK_DISPLAY} として読みます: {''.join(unknown)}）")
    if len(prompt) > model.config.block_size:
        print(f"（入力が長いので、最後の {model.config.block_size} 文字だけを読みます）")

    ids = tokenizer.encode(prompt or "\n")  # 空なら段落の頭から書き始める
    gen = model.generate(ids, style, temperature, top_k, banned=[tokenizer.unk_id], rng=rng)
    out = []
    for token_id in gen:
        ch = tokenizer.decode([token_id])
        out.append(ch)
        print(ch, end="", flush=True)
        # 指定の長さを超えたら、文の切れ目で止める（切れ目が来なければ +100 文字で打ち切り）
        if len(out) >= length and (ch in STOP_CHARS or len(out) >= length + 100):
            break
    print()
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="文豪風に文章の続きを生成する")
    parser.add_argument("--ckpt", type=Path, help="モデルのフォルダ（省略時は最後に学習したもの）")
    parser.add_argument("--prompt", help="書き出しの文章（省略すると対話モード）")
    parser.add_argument("--author", help="文体を真似る作家（複数の作家で学習した場合）")
    parser.add_argument("--length", type=int, default=200, help="生成するおおよその文字数")
    parser.add_argument("--temperature", type=float, default=0.8, help="小さいほど無難、大きいほど冒険的")
    parser.add_argument("--top-k", type=int, default=40, help="確率上位 k 文字の中から選ぶ")
    parser.add_argument("--seed", type=int, help="乱数の種（同じ値なら毎回同じ文章になる）")
    args = parser.parse_args()

    ckpt = args.ckpt or find_latest_checkpoint()
    model, tokenizer, authors = load_model(ckpt)
    rng = np.random.default_rng(args.seed)
    author = args.author or authors[0]
    if author not in authors:
        raise SystemExit(f"このモデルが知っている作家: {', '.join(authors)}")

    def run(prompt):
        write_continuation(
            model, tokenizer, prompt, authors.index(author), args.length, args.temperature, args.top_k, rng
        )

    if args.prompt is not None:
        print(args.prompt, end="")
        run(args.prompt)
        return

    print(f"文豪AI（モデル: {ckpt.name} / 作家: {', '.join(authors)}）")
    print("書き出しの文章を入力すると、その続きを書きます。")
    print("コマンド: /author 作家名  /temp 0.8  /len 200  /quit\n")
    while True:
        try:
            line = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/quit", "/exit"):
            break
        cmd, _, value = line.partition(" ")
        if cmd == "/author":
            if value in authors:
                author = value
                print(f"→ {author} 風に書きます")
            else:
                print(f"このモデルが知っている作家: {', '.join(authors)}")
            continue
        if cmd in ("/temp", "/len"):
            try:
                if cmd == "/temp":
                    args.temperature = float(value)
                else:
                    args.length = int(value)
                print(f"→ temperature={args.temperature}, length={args.length}")
            except ValueError:
                print(f"使い方: {cmd} 数値")
            continue

        print(f"{author}AI> {line}", end="")
        run(line)
        print()


if __name__ == "__main__":
    main()

"""ステップ3: モデルを学習させる。

使い方:
    uv run python train.py --preset tiny     # 動作確認用（数分）
    uv run python train.py                   # 標準設定（small）
    uv run python train.py --resume          # 中断した学習の続きから

Ctrl+C で中断しても、その時点までの状態が保存されるので --resume で再開できる。
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from numneuron.checkpoint import load_weights, save_config, save_weights
from numneuron.layers import cross_entropy
from numneuron.model import GPT, GPTConfig
from numneuron.optim import AdamW, clip_grad_norm, cosine_lr
from numneuron.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

# モデルの大きさと学習の長さの組み合わせ
PRESETS = {
    "tiny": dict(n_layer=2, n_head=4, n_embd=64, block_size=64, batch_size=32, max_iters=600, lr=2e-3),
    "small": dict(n_layer=4, n_head=4, n_embd=192, block_size=128, batch_size=32, max_iters=5000, lr=1e-3),
    "medium": dict(n_layer=6, n_head=6, n_embd=288, block_size=192, batch_size=32, max_iters=8000, lr=6e-4),
}
MODEL_KEYS = ("n_layer", "n_head", "n_embd", "block_size")


def get_batch(tokens, styles, block_size, batch_size, rng):
    """コーパスからランダムな位置の文章を batch_size 本切り出す。

    x: 入力の文字列、y: x を1文字ずらしたもの（各位置の「次の文字」= 正解）
    """
    T = block_size
    ix = rng.integers(0, len(tokens) - T - 1, size=batch_size)
    # 作家の境目をまたぐ切り出しは引き直す
    while (bad := styles[ix] != styles[ix + T]).any():
        ix[bad] = rng.integers(0, len(tokens) - T - 1, size=bad.sum())
    window = tokens[ix[:, None] + np.arange(T + 1)].astype(np.int64)
    return window[:, :-1], window[:, 1:], styles[ix].astype(np.int64)


def estimate_loss(model, tokens, styles, block_size, batch_size, n_batches):
    rng = np.random.default_rng(1234)  # 毎回同じ部分で測ると、前回と比べやすい
    losses = []
    for _ in range(n_batches):
        x, y, s = get_batch(tokens, styles, block_size, batch_size, rng)
        losses.append(cross_entropy(model.forward(x, s), y)[0])
    return float(np.mean(losses))


def sample_text(model, tokenizer, prompt, style, length=100):
    gen = model.generate(
        tokenizer.encode(prompt), style, banned=[tokenizer.unk_id], rng=np.random.default_rng(0)
    )
    return prompt + tokenizer.decode(next(gen) for _ in range(length))


def resolve_settings(args, ckpt_dir: Path) -> dict:
    if args.resume:
        saved = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))["train"]
        if args.max_iters is not None:
            saved["max_iters"] = args.max_iters
        return saved
    settings = dict(PRESETS[args.preset])
    for key in (*MODEL_KEYS, "batch_size", "max_iters", "lr"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    settings.update(
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        sample_prompt=args.sample_prompt,
    )
    return settings


def main():
    parser = argparse.ArgumentParser(description="文字単位 GPT を学習する")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--name", help="保存先 checkpoints/<name>（省略時はプリセット名）")
    parser.add_argument("--resume", action="store_true", help="保存済みの状態から学習を再開する")
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--lr", type=float, help="学習率の最大値")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-prompt", default="吾輩は", help="評価のたびに表示する生成例の書き出し")
    args = parser.parse_args()

    ckpt_dir = ROOT / "checkpoints" / (args.name or args.preset)
    s = resolve_settings(args, ckpt_dir)

    # --- データ ---
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    tokenizer = CharTokenizer(meta["itos"])
    authors = meta["authors"]
    data = np.load(DATA_DIR / "dataset.npz")
    train_tok, train_sty = data["train_tokens"], data["train_styles"]
    val_tok, val_sty = data["val_tokens"], data["val_styles"]

    # --- モデルと最適化手法 ---
    config = GPTConfig(vocab_size=len(tokenizer), n_styles=len(authors), **{k: s[k] for k in MODEL_KEYS})
    model = GPT(config, seed=s["seed"])
    params = model.parameters()
    optimizer = AdamW(params, weight_decay=s["weight_decay"])
    start_iter, best_val = 0, math.inf

    if args.resume:
        saved = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
        if saved["itos"] != tokenizer.itos or saved["authors"] != authors:
            raise SystemExit("prepare.py をやり直したため、保存済みモデルと文字の辞書が一致しません")
        state = load_weights(ckpt_dir / "state.npz", model)
        optimizer.load_state_dict(state)
        start_iter, best_val = int(state["iter"]), float(state["best_val"])
        print(f"{start_iter} 回目から再開します")
    save_config(ckpt_dir, config, tokenizer, authors, train=s)

    print(f"作家: {', '.join(authors)}")
    print(f"パラメータ数: {model.num_params():,}  設定: {config}")
    print(f"1回の学習で {s['batch_size']} 本 × {s['block_size']} 文字を読み、{s['max_iters']} 回くり返します")

    rng = np.random.default_rng(s["seed"] + start_iter)
    sample_style = 0

    def evaluate_and_save(it):
        nonlocal best_val
        val_loss = estimate_loss(model, val_tok, val_sty, s["block_size"], s["batch_size"], s["eval_batches"])
        mark = ""
        if val_loss < best_val:
            best_val = val_loss
            save_weights(ckpt_dir / "model.npz", model)
            mark = "（ベストを更新・保存）"
        save_weights(
            ckpt_dir / "state.npz", model, **optimizer.state_dict(),
            iter=np.array(it), best_val=np.array(best_val),
        )
        print(f"\n=== {it} 回目  検証用データの損失 {val_loss:.3f}（困惑度 {math.exp(val_loss):.1f}）{mark}")
        print("生成例: " + sample_text(model, tokenizer, s["sample_prompt"], sample_style).replace("\n", "⏎"))
        print()

    it = start_iter
    smooth_loss = None
    t0 = time.time()
    try:
        while it < s["max_iters"]:
            if it % s["eval_interval"] == 0 and it != start_iter:
                evaluate_and_save(it)

            lr = cosine_lr(it, s["lr"], s["warmup"], s["max_iters"])
            x, y, style = get_batch(train_tok, train_sty, s["block_size"], s["batch_size"], rng)

            # 学習の1ステップ: 順伝播 → 損失 → 逆伝播 → パラメータ更新
            logits = model.forward(x, style)
            loss, dlogits = cross_entropy(logits, y)
            model.zero_grad()
            model.backward(dlogits)
            grad_norm = clip_grad_norm(params, 1.0)
            optimizer.step(lr)
            it += 1

            smooth_loss = loss if smooth_loss is None else 0.95 * smooth_loss + 0.05 * loss
            if it % args.log_interval == 0:
                elapsed = time.time() - t0
                sec_per_iter = elapsed / (it - start_iter)
                eta = sec_per_iter * (s["max_iters"] - it)
                print(
                    f"{it:>6}/{s['max_iters']}  損失 {smooth_loss:.3f}  学習率 {lr:.1e}  "
                    f"勾配ノルム {grad_norm:.2f}  {sec_per_iter * 1000:.0f}ms/回  残り {eta / 60:.1f}分",
                    flush=True,
                )
        evaluate_and_save(it)
        print(f"学習完了。保存先: {ckpt_dir}")
    except KeyboardInterrupt:
        save_weights(
            ckpt_dir / "state.npz", model, **optimizer.state_dict(),
            iter=np.array(it), best_val=np.array(best_val),
        )
        print(f"\n中断しました（{it} 回目）。--resume で再開できます: {ckpt_dir}")


if __name__ == "__main__":
    main()

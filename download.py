"""ステップ1: 青空文庫から作家の作品テキストをダウンロードする。

使い方:
    uv run python download.py                     # 夏目漱石（デフォルト）
    uv run python download.py 芥川竜之介 太宰治    # 他の文豪（姓名を続けて書く）

保存先: data/raw/<作家名>/<作品ID>_<作品名>.txt （青空文庫の書式のまま UTF-8 で保存）
"""

import argparse
import csv
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CATALOG_PATH = ROOT / "data" / "aozora_catalog.zip"
CATALOG_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
USER_AGENT = "nastume_neuron/0.1 (personal study project)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as res:
        return res.read()


def load_catalog() -> list[dict[str, str]]:
    """青空文庫の全作品一覧（CSV）を読み込む。初回だけダウンロードして保存しておく。"""
    if not CATALOG_PATH.exists():
        print("青空文庫の作品一覧をダウンロード中...")
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_bytes(fetch(CATALOG_URL))
    with zipfile.ZipFile(CATALOG_PATH) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        text = zf.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def find_works(catalog, author: str, all_kana: bool) -> list[dict[str, str]]:
    works, seen = [], set()
    for row in catalog:
        if row["姓"] + row["名"] != author or row["役割フラグ"] != "著者":
            continue
        if row["作品著作権フラグ"] != "なし" or not row["テキストファイルURL"].endswith(".zip"):
            continue
        # 旧仮名づかい（「云ふ」など）が混ざると学習がばらつくので、標準では新字新仮名だけ使う
        if not all_kana and row["文字遣い種別"] != "新字新仮名":
            continue
        if row["作品ID"] in seen:
            continue
        seen.add(row["作品ID"])
        works.append(row)
    return works


def extract_text(zip_bytes: bytes) -> str:
    """zip の中の .txt を取り出す。青空文庫のテキストは Shift_JIS で書かれている。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        txt = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
        if txt is None:
            raise ValueError("zip 内に .txt がありません")
        return zf.read(txt).decode("cp932", errors="replace")


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]', "_", name)


def main():
    parser = argparse.ArgumentParser(description="青空文庫から作品をダウンロードする")
    parser.add_argument("authors", nargs="*", default=["夏目漱石"], help="作家名（例: 夏目漱石 芥川竜之介）")
    parser.add_argument("--all-kana", action="store_true", help="旧字・旧仮名の作品も含める")
    parser.add_argument("--interval", type=float, default=1.0, help="ダウンロードの間隔（秒）。サーバーに負荷をかけないため")
    args = parser.parse_args()

    catalog = load_catalog()
    for author in args.authors:
        works = find_works(catalog, author, args.all_kana)
        if not works:
            print(f"「{author}」の作品が見つかりませんでした（姓と名を続けて、青空文庫の表記で指定してください）")
            continue

        out_dir = RAW_DIR / author
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{author}: {len(works)} 作品")
        for i, row in enumerate(works, 1):
            path = out_dir / f"{row['作品ID']}_{safe_filename(row['作品名'])}.txt"
            if path.exists():
                continue
            try:
                text = extract_text(fetch(row["テキストファイルURL"]))
            except Exception as e:
                print(f"  [{i}/{len(works)}] 失敗: {row['作品名']} ({e})")
                continue
            path.write_text(text, encoding="utf-8")
            print(f"  [{i}/{len(works)}] {row['作品名']}（{len(text):,} 文字）")
            time.sleep(args.interval)

    print(f"保存先: {RAW_DIR}")


if __name__ == "__main__":
    main()

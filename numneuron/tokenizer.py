"""文字単位のトークナイザ。日本語は1文字ずつ区切っても意味のある単位になりやすい。"""

from collections import Counter
from collections.abc import Iterable


class CharTokenizer:
    UNK = "<unk>"  # 辞書にない文字。ID は 0
    UNK_DISPLAY = "〓"  # 表示するときの代わりの文字（青空文庫でも使われる「げた記号」）

    def __init__(self, itos: list[str]):
        self.itos = itos  # ID → 文字
        self.stoi = {ch: i for i, ch in enumerate(itos)}  # 文字 → ID

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 3) -> "CharTokenizer":
        """コーパスに min_count 回以上出てくる文字で辞書を作る。まれな文字は <unk> にまとめる。"""
        counts = Counter()
        for text in texts:
            counts.update(text)
        chars = sorted(ch for ch, n in counts.items() if n >= min_count)
        return cls([cls.UNK, *chars])

    @property
    def unk_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.UNK_DISPLAY if i == self.unk_id else self.itos[i] for i in ids)

    def unknown_chars(self, text: str) -> list[str]:
        """辞書にない文字の一覧（入力チェック用）。"""
        return sorted({ch for ch in text if ch not in self.stoi})

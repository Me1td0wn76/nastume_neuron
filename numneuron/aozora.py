"""青空文庫のテキストファイルから本文だけを取り出す。

青空文庫の .txt には本文以外に以下が含まれるので取り除く:
- 冒頭: 題名・著者名と「テキスト中に現れる記号について」の説明
- 末尾: 「底本：」から始まる書誌情報
- ルビ:   漢字《かんじ》、｜（ルビの始まりの印）
- 注記:   ［＃「一」は中見出し］、※［＃「土へん＋竒」、第3水準1-15-51］ など
"""

import re

_SEPARATOR = re.compile(r"^-{20,}\s*$", re.MULTILINE)
_FOOTER = re.compile(r"^底本：", re.MULTILINE)
_HEADING_LINE = re.compile(r"^.*［＃「[^」]*」は[大中小]?見出し］.*$", re.MULTILINE)
_RUBY = re.compile(r"《[^》]*》")
_NOTE = re.compile(r"※?［＃[^］]*］")
_CHAPTER_NUMBER = re.compile(r"^[（(]?[一二三四五六七八九十百〇0-9０-９]+[）)]?$")


def clean_aozora(raw: str) -> str:
    text = raw.replace("\r\n", "\n")

    # 冒頭: 「-----」の区切り線が2本あれば、その間が記号の説明
    seps = list(_SEPARATOR.finditer(text))
    if len(seps) >= 2:
        text = text[seps[1].end() :]
    else:
        text = text.split("\n", 2)[-1]  # 題名と著者名の2行だけ落とす

    footer = _FOOTER.search(text)
    if footer:
        text = text[: footer.start()]

    text = _HEADING_LINE.sub("", text)  # 「一」「二」などの章見出しの行
    text = _RUBY.sub("", text)
    text = text.replace("｜", "")
    text = _NOTE.sub("", text)
    text = text.replace("※", "")

    # 段落頭の全角スペースを取り、空行と章番号だけの行をなくす（1行 = 1段落 になる）
    lines = (line.strip(" 　\t") for line in text.split("\n"))
    return "\n".join(line for line in lines if line and not _CHAPTER_NUMBER.match(line))

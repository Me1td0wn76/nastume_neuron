from numneuron.aozora import clean_aozora
from numneuron.tokenizer import CharTokenizer

SAMPLE = """坊っちゃん\r
夏目漱石\r
\r
-------------------------------------------------------\r
【テキスト中に現れる記号について】\r
\r
《》：ルビ\r
（例）坊《ぼ》っちゃん\r
-------------------------------------------------------\r
\r
［＃５字下げ］一［＃「一」は中見出し］\r
\r
　親譲《おやゆず》りの無鉄砲《むてっぽう》で小供の時から損ばかりしている。\r
　ある日の夕方｜折戸《おりど》の蔭《かげ》に隠れた。※［＃「土へん＋竒」、第3水準1-15-51］\r
［＃地から１字上げ］（明治三十九年四月）\r
\r
\r
\r
底本：「ちくま日本文学全集　夏目漱石」筑摩書房\r
入力：真先芳秋\r
"""


def test_clean_aozora_keeps_only_body():
    assert clean_aozora(SAMPLE) == (
        "親譲りの無鉄砲で小供の時から損ばかりしている。\n"
        "ある日の夕方折戸の蔭に隠れた。\n"
        "（明治三十九年四月）"
    )


def test_tokenizer_roundtrip_and_unknown():
    tok = CharTokenizer.build(["吾輩は猫である。", "吾輩は"], min_count=1)
    ids = tok.encode("吾輩は犬")
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "吾輩は" + CharTokenizer.UNK_DISPLAY
    assert tok.unknown_chars("吾輩は犬である") == ["犬"]

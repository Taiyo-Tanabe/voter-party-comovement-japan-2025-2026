"""国会議員などの「氏名 → 名字」の対応表を、会議録データから作る"""
import re

import pandas as pd

# 見出し「○長妻委員」「○赤澤国務大臣」「○あべ委員」から名字を取り出す
HEAD = re.compile(
    r"^○([一-龥々ヶぁ-んァ-ヶー]{1,6})(?:（[^）]*）)?"
    r"(?:委員|君|国務大臣|内閣総理大臣|副大臣|大臣政務官|政府特別補佐人)"
)
KANA_HEAD = re.compile(r"^[ぁ-んァ-ヶー]+")


def guess_surname(name: str):
    """名字が分からない人の名字を推測する(方法C)"""
    m = KANA_HEAD.match(name)
    if m and len(m.group()) < len(name):          # 「いんどう周作」→「いんどう」
        return m.group()
    if len(name) >= 3:                             # 漢字の氏名は先頭2文字(「々」「ヶ」が2文字目なら3文字)
        return name[:3] if name[1] in "々ヶ" else name[:2]
    return None


def build_member_table(df: pd.DataFrame) -> pd.DataFrame:
    """除外する前の全データ(df)から、氏名と名字の対応表を作る

    source 列: header(見出しから)/ known(他の議員の名字で区切った)/ guess(推測)/ none(名字なし)
    """
    names = [n for n in df["speaker"].dropna().unique() if n != "会議録情報" and len(n) >= 2]

    # ① 見出しから(氏名まるごとが取れた行は除く)
    sur = df["speech"].str.extract(HEAD)[0]
    pairs = pd.DataFrame({"speaker": df["speaker"], "surname": sur}).dropna().drop_duplicates()
    pairs = pairs[[n.startswith(s) and n != s for n, s in zip(pairs["speaker"], pairs["surname"])]]
    pairs = pairs.drop_duplicates("speaker")
    found = dict(zip(pairs["speaker"], pairs["surname"]))

    # ② すでに分かっている名字で区切る(長い名字から試す)
    known = sorted(set(found.values()), key=len, reverse=True)

    rows = []
    for name in names:
        if name in found:
            rows.append((name, found[name], "header"))
            continue
        s = next((k for k in known if name.startswith(k) and len(k) < len(name)), None)
        if s:
            rows.append((name, s, "known"))
            continue
        # ③ 推測
        g = guess_surname(name)
        rows.append((name, g, "guess" if g else "none"))

    return pd.DataFrame(rows, columns=["full_name", "surname", "source"])
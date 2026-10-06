"""国会会議録の発言から、政党の位置の推定に使わない発言を除き、本文を整える"""
import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# 1. 除外ルール
# ---------------------------------------------------------------
# 委員長などの議事進行の見出し
#   衆議院: 「○安住委員長　」「○村井小委員長　」
#   参議院: 「○委員長（鶴保庸介君）　」
CHAIR = r"^○\S*?(?:委員長|議長|会長|主査|座長|理事)(?:代理)?(?:（[^）]*）)?[\s　]"

# 委員長報告・分科会報告・派遣報告(最初の行で「御報告申し上げます/いたします」と宣言する)
#   本会議の委員長報告は見出しが「○阿久津幸彦君」なので、CHAIR では拾えない
REPORT = r"^[^\n]{0,200}?御報告(?:申し上げ|いたし)"


def filter_speeches(df: pd.DataFrame) -> pd.DataFrame:
    """除外ルールを順に適用し、残った発言を返す"""
    df = df[df["speechOrder"] > 0]                    # 会議録の冒頭(出席者一覧)
    df = df[df["speakerPosition"].isna()]             # 大臣・政務官・官僚・議長(答弁側)
    df = df[df["speakerRole"].isna()]                 # 参考人・証人など
    df = df[~df["speech"].str.match(CHAIR)]           # 委員長などの議事進行
    df = df[~df["speech"].str.contains(REPORT)]       # 委員長報告などの報告
    df = df[df["speakerGroup"].notna()]               # 会派不明
    return df.copy()


# ---------------------------------------------------------------
# 2. 本文の整形
# ---------------------------------------------------------------
# 発言の先頭の見出し(「○長妻委員　」「○竹谷とし子君　」)
HEADER = r"^○\S+?[\s　]"


def clean(text: str) -> str:
    """本文をBERTに入れる前に整える"""
    text = re.sub(HEADER, "", text)                   # 先頭の見出し
    text = unicodedata.normalize("NFKC", text)        # 全角英数字・記号を半角にそろえる(ＮＨＫ→NHK)
    text = re.sub(r"[―─]{3,}", "", text)              # 区切り線
    text = re.sub(r"\s+", " ", text)                  # 改行・空白をまとめる
    return text.strip()


# ---------------------------------------------------------------
# 3. 党名のマスキング(全員の発言に適用)
# ---------------------------------------------------------------
MASK = "政党"

PARTY_NAMES = [
    # 会派名
    "自由民主党・無所属の会", "立憲民主・社民・無所属", "立憲民主党・無所属", "立憲民主・無所属",
    "国民民主党・無所属クラブ", "国民民主党・新緑風会", "中道改革連合・無所属",
    "チームみらい・無所属の会", "自由民主党・国民の声",
    "有志の会", "沖縄の風", "改革の会", "減税保守こども",
    # 政党名
    "自由民主党", "立憲民主党", "日本維新の会", "公明党", "日本共産党",
    "国民民主党", "れいわ新選組", "中道改革連合", "社会民主党", "参政党", "日本保守党",
    "チームみらい", "NHKから国民を守る党", "NHK党",
    # 略称
    "自民党", "立憲民主", "国民民主", "共産党", "社民党", "維新の会",
    "自民", "立憲", "維新", "公明", "共産", "れいわ", "社民",
    "N国党", "N国", "N党",
    # 会派名の一部だけで使われる語
    "新緑風会", "無所属クラブ",
    # 組み合わせの略称
    "自公", "自維",
]
# 長い名前から順に並べ替える(「立憲民主党」を「立憲」より先に置き換えるため)
PARTY_NAMES = sorted(PARTY_NAMES, key=len, reverse=True)

# 党名以外の語に含まれる場合は置き換えない(例: 公明正大)
NOT_PARTY = {"公明": "(?!正大)"}

PARTY_RE = re.compile("|".join(
    re.escape(name) + NOT_PARTY.get(name, "") for name in PARTY_NAMES
))


def mask_parties(text: str) -> str:
    """党名をすべて同じ語に置き換える"""
    return PARTY_RE.sub(MASK, text)


# 一般の言葉と重なる党の略称は、党を指す文脈のときだけマスクする
AMBIG = "中道|国民|参政|みらい"
# 後ろの文脈で判断:「中道提出」「国民さん」「中道の小川代表」など
CTX_AFTER = re.compile(rf"({AMBIG})(?=提出|さん|案|の[一-龥]{{1,4}}(?:代表|幹事長|幹事|議員|委員|提出者))")
# 並びで判断:「政党、中道」「中道、政党」
LIST_BEFORE = re.compile(rf"(?<=政党、)({AMBIG})")
LIST_AFTER = re.compile(rf"({AMBIG})(?=、政党)")


def mask_ambiguous(text: str) -> str:
    """一般の言葉と重なる党の略称を、党を指す文脈のときだけマスクする"""
    text = CTX_AFTER.sub(MASK, text)
    for _ in range(5):                    # 「政党、中道、国民、参政」のような連続に対応するため、くり返す
        new = LIST_AFTER.sub(MASK, LIST_BEFORE.sub(MASK, text))
        if new == text:
            break
        text = new
    return text


# ---------------------------------------------------------------
# 4. 発言者本人に関わるマスキング(その人の発言にだけ適用)
# ---------------------------------------------------------------
# 自分の党を指す略称のうち、一般の言葉と重なるもの
OWN_ALIASES = {"中道": ["中道"]}


def mask_own(text: str, speaker: str, party: str) -> str:
    """本人の名前と、自分の党の略称をマスクする"""
    if speaker:
        text = text.replace(speaker, "議員")
    for alias in OWN_ALIASES.get(party, []):
        text = text.replace(alias, MASK)
    return text


# ---------------------------------------------------------------
# 5. 国会議員などの名前のマスキング
# ---------------------------------------------------------------
# 対応表(氏名・名字)は src/preprocess/members.py の build_member_table で作り、CSVに保存しておく
MEMBERS_CSV = Path(__file__).resolve().parents[2] / "data" / "external" / "member_names.csv"

# 名字を消すのは、後ろにこれらの役職が続くときだけ(「林業」「平和」などを壊さないため)
TITLES = "総理|大臣|副大臣|官房長官|長官|政務官|代表|幹事長|さん|氏|議員|委員|君|先生"

# 名字の一覧に入っていても、一般の言葉として使われるので消さない語
NOT_SURNAMES = {"今"}


def _alternation(words):
    """長い順に並べて「|」でつなぐ"""
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


def _load_name_patterns():
    if not MEMBERS_CSV.exists():
        warnings.warn(f"{MEMBERS_CSV} がないため、議員名のマスクを行いません")
        return None, None
    m = pd.read_csv(MEMBERS_CSV)
    full_names = m["full_name"].dropna().tolist()
    surnames = [s for s in m["surname"].dropna() if s not in NOT_SURNAMES]   # ← 追加
    # 名字を推測した人は、先頭3文字も名字の候補にする(長谷部・小野寺など)
    surnames += [n[:3] for n in m.loc[m["source"] == "guess", "full_name"].dropna() if len(n) >= 4]
    full_re = re.compile(_alternation(full_names))
    sur_re = re.compile(rf"(?<![一-龥])(?:{_alternation(surnames)})(?=(?:前|元)?(?:{TITLES}))")
    return full_re, sur_re


FULL_NAME_RE, SURNAME_RE = _load_name_patterns()


def mask_person_names(text: str) -> str:
    """氏名はどこでも消し、名字は役職の前にあるときだけ消す(「高市総理」→「総理」)"""
    if FULL_NAME_RE is None:
        return text
    text = FULL_NAME_RE.sub("", text)
    return SURNAME_RE.sub("", text)


# ---------------------------------------------------------------
# 6. まとめ役
# ---------------------------------------------------------------

# 追加:「政党の議員です。」「政党、議員でございます。」のような自己紹介
SELF_INTRO = re.compile(r"政党[の、]?議員(?:でございます|です)。?")

def prepare_text(speech, speaker=None, party=None, mask=True):
    text = clean(speech)
    if not mask:
        return text
    text = mask_parties(text)                # ← 先に党名
    text = mask_own(text, speaker, party)    # ← 後に本人の名前・自党の略称
    text = mask_ambiguous(text)
    text = mask_person_names(text)
    text = SELF_INTRO.sub("", text)           # 追加:自己紹介の文を消す
    text = text.replace("我々政党", "我々")    # 追加:「我々日本維新の会は」→「我々は」
    return text
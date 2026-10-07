"""生の会議録JSONから、分析用の発言データ(parquet)を作る

使い方(プロジェクトのルートで):
    python scripts/02_preprocess.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess.filters import filter_speeches, prepare_text

RAW = ROOT / "data" / "raw" / "kokkai"
EXT = ROOT / "data" / "external"
GROUP_MAP = EXT / "group_party_map.csv"               # 会派 → 党
OVERRIDE = EXT / "speaker_party_override.csv"         # 合同会派にいた他党の議員 → 本人の党
EXCLUDE = EXT / "speaker_exclude.csv"                 # 議員でない人(公述人など)
OUT = ROOT / "data" / "processed" / "speeches.parquet"

# 手続きの発言(緊急動議・附帯決議案などの読み上げ)。立場を含まないので除外する
# 「附帯決議案」「緊急動議」という言葉だけでは消さない(討論や質問の中にも出てくるため)
PROC = r"緊急動議を提出|趣旨の説明に代え|趣旨の説明といたします|案文を朗読|案文の朗読|案文朗読|附帯決議案を提出"
# ただし「会派を代表」して述べる趣旨説明(予算の編成替え動議、解任決議案など)は党の立場なので残す
KEEP = r"会派を代表"

# 保存する列(元の21列のうち、分析に使うものと、確認に使うもの)
COLUMNS = ["speechID", "issueID", "date", "session", "nameOfHouse", "nameOfMeeting",
           "speechOrder", "speaker", "speakerGroup", "party",
           "text", "text_nomask", "speechURL"]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """月ごとのJSONをすべて読み込み、1つの表にする"""
    dfs = []
    for p in sorted(raw_dir.glob("speech_*.json")):
        with open(p, encoding="utf-8") as f:
            records = json.load(f).get("speechRecord", [])   # 0件の月はキーがない
        dfs.append(pd.DataFrame(records))
    return pd.concat(dfs, ignore_index=True)


def assign_party(q: pd.DataFrame) -> pd.DataFrame:
    """会派から党を決め、合同会派にいた他党の議員だけ本人の党に上書きする"""
    gmap = pd.read_csv(GROUP_MAP, encoding="utf-8")
    q["party"] = q["speakerGroup"].map(dict(zip(gmap["speakerGroup"], gmap["party"]))).fillna("その他")
    print("\n「その他」になった会派:")
    print(q.loc[q["party"] == "その他", "speakerGroup"].value_counts().to_string())

    ov = pd.read_csv(OVERRIDE, encoding="utf-8")
    ov_map = dict(zip(ov["speaker"], ov["party"]))
    m = q["speaker"].isin(ov_map.keys())
    q.loc[m, "party"] = q.loc[m, "speaker"].map(ov_map)
    print("\n党を上書き:", int(m.sum()), "件")
    return q


def drop_rows(q: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    """mask が True の行を消し、消した件数を表示する"""
    print(f"{label}:", int(mask.sum()), "件")
    return q[~mask].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-08-01", help="この日より前の発言だけを使う")
    args = ap.parse_args()
    t0 = time.time()

    # 1. 読み込み
    df = load_raw(RAW)
    df["date"] = pd.to_datetime(df["date"])
    print("読み込み:", len(df), "件")

    # 2. 除外(議長・報告などのルール、期間、非議員、手続きの発言)
    q = filter_speeches(df)
    q = q[q["date"] < args.end].copy()
    print("ルールと期間で除外した後:", len(q), "件")
    ex = pd.read_csv(EXCLUDE, encoding="utf-8")
    q = drop_rows(q, q["speaker"].isin(ex["speaker"]), "非議員を除外")
    q = drop_rows(
        q, q["speech"].str.contains(PROC, na=False)
        & ~q["speech"].str.contains(KEEP, na=False), "手続きの発言を除外"
    )

    # 3. 党を決める(会派 → 党、合同会派の他党議員は上書き)
    #    マスクで「自分の党の名前」を使うので、本文の整形より前に決めておく
    q = assign_party(q)

    # 4. 本文の整形とマスク(ここが一番時間がかかる)
    print("\n本文を整形しています…")
    q["text"] = [prepare_text(s, sp, p) for s, sp, p in zip(q["speech"], q["speaker"], q["party"])]
    q["text_nomask"] = [prepare_text(s, mask=False) for s in q["speech"]]

    # 5. 保存
    OUT.parent.mkdir(parents=True, exist_ok=True)
    q[COLUMNS].to_parquet(OUT, index=False)
    print(f"\n保存しました: {OUT}")
    print(f"{len(q)}件, {time.time() - t0:.0f}秒")
    print(q["party"].value_counts().to_string())


if __name__ == "__main__":
    main()
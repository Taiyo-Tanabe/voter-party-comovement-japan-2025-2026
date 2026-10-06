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
GROUP_MAP = ROOT / "data" / "external" / "group_party_map.csv"
OUT = ROOT / "data" / "processed" / "speeches.parquet"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-08-01", help="この日より前の発言だけを使う")
    args = ap.parse_args()
    t0 = time.time()

    # 1. 読み込み
    df = load_raw(RAW)
    df["date"] = pd.to_datetime(df["date"])
    print("読み込み:", len(df), "件")

    # 2. 除外
    q = filter_speeches(df)
    q = q[q["date"] < args.end].copy()
    print("除外後:", len(q), "件")

    # 3. 会派 → 党
    gmap = pd.read_csv(GROUP_MAP, encoding="utf-8")
    q["party"] = q["speakerGroup"].map(dict(zip(gmap["speakerGroup"], gmap["party"]))).fillna("その他")
    print("\n「その他」になった会派:")
    print(q.loc[q["party"] == "その他", "speakerGroup"].value_counts().to_string())

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
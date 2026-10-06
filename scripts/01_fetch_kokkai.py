"""使い方(プロジェクトのルートで実行):
    python scripts/01_fetch_kokkai.py                          # 2025-01〜2026-07
    python scripts/01_fetch_kokkai.py --start 2026-08 --end 2026-08
    python scripts/01_fetch_kokkai.py --start 2025-03 --end 2025-03 --force   # 取り直し
"""
import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]   # scripts/ の1つ上 = プロジェクトのルート
sys.path.insert(0, str(ROOT))
from src.collect.kokkai import fetch_month, save_month  # noqa: E402

OUT = ROOT / "data" / "raw" / "kokkai"


def months(start, end):
    """'2025-01'〜'2026-07' のような範囲の (年, 月) を順に返す"""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01")
    p.add_argument("--end", default="2026-07")
    p.add_argument("--force", action="store_true", help="取得済みの月も取り直す")
    args = p.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "survey_on_party_systems (academic research)"

    for y, m in months(args.start, args.end):
        path = OUT / f"speech_{y}-{m:02d}.json"
        if path.exists() and not args.force:
            print(f"skip  {path.name}(取得済み)")
            continue
        print(f"fetch {y}-{m:02d}")
        result = fetch_month(y, m, session)
        got, total = len(result["speechRecord"]), result["numberOfRecords"]
        if got != total:
            print(f"  warning: 件数が一致しない {got}/{total}")
        print(f"  saved {save_month(result, OUT).name}")


if __name__ == "__main__":
    main()
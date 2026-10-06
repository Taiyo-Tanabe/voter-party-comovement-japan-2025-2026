"""国会会議録検索システム API(発言単位)から月ごとに全件取得する"""
import json
import time
import calendar
from datetime import datetime
from pathlib import Path

import requests

URL = "https://kokkai.ndl.go.jp/api/speech"
PER_PAGE = 100     # 発言検索の1回あたりの上限
INTERVAL = 2.0     # リクエストの間隔(秒)。並列にはしない

def _get(session, params, retries=5):
    """1ページ分を取得。失敗したら待ち時間を倍々に延ばして再試行する"""
    for i in range(retries):
        try:
            r = session.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"\n  HTTP {r.status_code}: {r.text[:200]}")
        except (requests.RequestException, ValueError) as e:
            print(f"\n  error: {e}")
        wait = INTERVAL * 2 ** (i + 1)          # 4, 8, 16, 32, 64秒
        print(f"  retry {i + 1}/{retries} in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"取得に失敗: {params}")

def fetch_month(year, month, session=None):
    """year年month月の発言を全件取得して辞書で返す"""
    session = session or requests.Session()
    last = calendar.monthrange(year, month)[1]
    params = {
        "from": f"{year}-{month:02d}-01", 
        "until": f"{year}-{month:02d}-{last}", 
        "maximumRecords": PER_PAGE, 
        "recordPacking": "json", 
        "startRecord": 1
    }
    records, total = [], 0
    while True:
        data = _get(session, params)
        total = data.get("numberOfRecords", 0)
        records += data.get("speechRecord", [])   # 0件のときはキー自体が無い
        print(f" {year}-{month:02d}: {len(records)}/{total}", end="\r")
        time.sleep(INTERVAL)
        nxt = data.get("nextRecordPosition")
        if not nxt:
            break
        params["startRecord"] = nxt
    print()
    return {
        "year": year, 
        "month": month, 
        "numberOfRecords": total, 
        "fetched_at": datetime.now().isoformat(timespec="seconds"), 
        "speechRecord": records
    }

def save_month(result, out_dir):
    """一時ファイルに書いてから名前を変える(途中で止まっても壊れたファイルを残さない)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"speech_{result['year']}-{result['month']:02d}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    tmp.replace(path)
    return path
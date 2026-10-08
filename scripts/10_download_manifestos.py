"""manifesto_sources.csv の url から公約をダウンロードして file の場所に保存する。

使い方(プロジェクトのフォルダで):
    python scripts/10_download_manifestos.py          # まだ無いファイルだけ取る
    python scripts/10_download_manifestos.py --force  # 全部取り直す
"""
import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "external" / "manifesto_sources.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-download"


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        final = r.geturl()
    # HTML が別のページ(トップページなど)に転送されたら、目的のページではない可能性が高い
    # (PDF は保存場所の移動で転送されることがあるので、下の %PDF の確認で判断する)
    if path.suffix == ".html" and final.rstrip("/") != url.rstrip("/"):
        raise ValueError(f"別のページに転送された → {final}")
    # PDF のはずなのに中身が PDF でない(エラーページなど)なら保存しない
    if path.suffix == ".pdf" and not data.startswith(b"%PDF"):
        raise ValueError("PDF ではないものが返ってきた(リンク切れの可能性)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="既にあるファイルも取り直す")
    args = ap.parse_args()

    src = pd.read_csv(SOURCES, dtype=str).fillna("")
    ok, skip, fail = 0, 0, []
    for r in src.itertuples():
        path = ROOT / r.file
        label = f"{r.election} {r.party} {path.name}"
        if not r.url:
            print(f"[URLなし] {label}  ← WARP などで手で保存する")
            skip += 1
            continue
        if path.exists() and not args.force:
            print(f"[あり]   {label}")
            skip += 1
            continue
        try:
            n = download(r.url, path)
            print(f"[保存]   {label}  {n / 1e6:.1f}MB")
            ok += 1
        except Exception as e:
            print(f"[失敗]   {label}  {e}")
            fail.append(label)
        time.sleep(1)  # 相手のサイトに負担をかけないよう1秒あける

    print(f"\n保存 {ok} / とばした {skip} / 失敗 {len(fail)}")
    for f in fail:
        print("  失敗:", f)


if __name__ == "__main__":
    main()

"""重点公約の全ページを、Claude で「1行 = 1政策」の表にする(13_segment の本番)。

方法は 13_segment_eval.py で比べて決めたもの(Claude Sonnet 5・effort medium)をそのまま使う。
プロンプト・Claude の呼び方・答えの読み方は 13_segment_eval.py から読み込み、採点した方法と本番がずれないようにする。

使い方(プロジェクトのフォルダで):
    python scripts/13_segment.py                 # 料金の見積もりを出すだけ(Claude は呼ばない)
    python scripts/13_segment.py --limit 5 --run # 最初の5ページだけ試す(出力の表もその5ページ分だけになる)
    python scripts/13_segment.py --run           # 全ページ(答えを保存済みのページは呼ばない。途中で止めても続きから)

出力:
    data/interim/manifestos/policies.parquet   1行 = 1政策の表
    reports/segment_check.csv                  ページごとの点検表(目で確かめるページを探す)
    data/interim/manifestos/segment/claude_medium/   ページごとの Claude の答えと calls.csv(料金の記録)

政策が無いと目で確かめたページ(表紙など。Claude が区切る代わりに返事を書いたもの)は
data/external/segment_skip_pages.csv に理由と一緒に書き、政策0行として扱う。
"""
import argparse
import importlib
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ev = importlib.import_module("13_segment_eval")   # 採点に使ったものと同じ PROMPT・seg_claude・parse_items を使う

ROOT = ev.ROOT
EFFORT = "medium"                                  # 13_segment_eval.py で比べて決めた(high と同じ F1 で料金は約3分の2)
BASE = ROOT / "data" / "interim" / "manifestos" / "segment" / "claude"   # 実際の保存先は claude_medium
ANSWERS = BASE.with_name(f"{BASE.name}_{EFFORT}")
EVAL_ANSWERS = ev.CLAUDE_DIR.with_name(f"{ev.CLAUDE_DIR.name}_{EFFORT}")  # 採点のときの答え(同じ設定なので使い回す)
OUT = ROOT / "data" / "interim" / "manifestos" / "policies.parquet"
CHECK = ROOT / "reports" / "segment_check.csv"
MIN_CHARS = 20                                     # これより短いページ(表紙など)は区切らない
SKIP = ROOT / "data" / "external" / "segment_skip_pages.csv"   # 政策が無いと目で確かめたページ(政策0行として扱う)
# Claude が区切る代わりに、こちらへの返事を書いたときに出やすい言い回し(点検表で「要確認(返事)」にする)
REPLY = r"文章が提供|提供された文章|申し訳|書き直す|書き直せ|貼り付け|お送りください|お送りいただ|政策内容なし|判断されます"
USD_PER_1000_CHARS = 0.034                         # 採点の6ページ(medium)の料金 $0.204 ÷ 6,088字 から


def answer_path(key):
    return ANSWERS / f"{key[0]}_{key[1]}_p{key[2]:03d}.txt"


def has_answer(key):
    p = answer_path(key)
    return p.exists() and p.read_text(encoding="utf-8").strip() != ""


def flat(text):
    """作文の確認用: 本文の空白・改行をすべて消して1本につなぐ(13_segment_eval.py の score() と同じ)"""
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Claude を呼んで区切る(付けなければ見積もりだけ)")
    ap.add_argument("--limit", type=int, help="最初の N ページだけ")
    args = ap.parse_args()

    pages = pd.read_parquet(ev.PAGES)
    pages = pages[pages["kind"] == "main"].copy()
    pages["text"] = pages["text"].fillna("")
    pages = pages.sort_values(["election", "doc", "page"]).reset_index(drop=True)
    short = pages["text"].str.len() < MIN_CHARS
    skip = pd.read_csv(SKIP) if SKIP.exists() else pd.DataFrame(columns=["election", "doc", "page", "reason"])
    skip_keys = set(zip(skip["election"], skip["doc"], skip["page"].astype(int)))
    todo = pages[~short]
    if args.limit:
        todo = todo.head(args.limit)

    # 採点のときの答え(同じ effort)があれば、本番のフォルダに写して使い回す
    ANSWERS.mkdir(parents=True, exist_ok=True)
    copied = 0
    for r in todo.itertuples():
        key = (r.election, r.doc, r.page)
        src = EVAL_ANSWERS / answer_path(key).name
        if not has_answer(key) and src.exists() and src.read_text(encoding="utf-8").strip():
            shutil.copy(src, answer_path(key))
            copied += 1

    need = todo[[(r.election, r.doc, r.page) not in skip_keys and not has_answer((r.election, r.doc, r.page))
                 for r in todo.itertuples()]]
    chars = int(need["text"].str.len().sum())
    print(f"重点公約 {len(pages)} ページ(20字未満 {int(short.sum())} ページは飛ばす)→ 対象 {len(todo)} ページ"
          f"(うち政策が無いページの一覧に入っているもの {len(skip_keys)})")
    print(f"  答えがあるページ {len(todo) - len(need)}(うち採点の答えを使い回し {copied})/ これから呼ぶページ {len(need)}")
    print(f"  料金の見積もり: 約 ${chars / 1000 * USD_PER_1000_CHARS:.2f}({chars:,} 字。失敗の呼び直しは含まない)")
    if not args.run:
        print("\n見積もりだけ表示しました。Claude を呼ぶには --run を付けてください。")
        return

    rows, checks = [], []
    for i, r in enumerate(todo.itertuples(), 1):
        key = (r.election, r.doc, r.page)
        if key in skip_keys:                       # 政策が無いと確かめたページは、Claude の答えを使わない
            items, status = [], "政策なし(一覧)"
        else:
            status = "保存済み" if has_answer(key) else "呼んだ"
            if status == "呼んだ":
                print(f"[{i}/{len(todo)}]", end="")
            items = ev.seg_claude(r.text, r.doc, key, effort=EFFORT, base=BASE)
            if status == "呼んだ" and not has_answer(key):
                status = "失敗"                    # 答えが空・上限で打ち切り(保存されていない → 次の実行で呼び直す)
            elif any(re.search(REPLY, it["raw"]) for it in items):
                status = "要確認(返事)"            # 区切る代わりに返事を書いた疑い → 目で見て、一覧に入れるか決める
        body = flat(r.text)
        made_up = [it["raw"] for it in items if it["text"] not in body]
        for n, it in enumerate(items, 1):
            rows.append({"election": r.election, "party": r.party, "doc": r.doc, "page": r.page, "item": n,
                         "heading": it["heading"], "text": it["raw"], "made_up": it["text"] not in body})
        checks.append({"election": r.election, "party": r.party, "doc": r.doc, "page": r.page,
                       "n_chars": len(r.text), "status": status, "n_items": len(items), "n_made_up": len(made_up),
                       "made_up_examples": " / ".join(m[:40] for m in made_up[:3])})

    pol = pd.DataFrame(rows)
    chk = pd.DataFrame(checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pol.to_parquet(OUT, index=False)
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    chk.to_csv(CHECK, index=False, encoding="utf-8")

    print(f"\n政策 {len(pol)} 行 / ページ {len(chk)}")
    print(chk["status"].value_counts().to_string())
    bad = chk[chk["status"].isin(["失敗", "要確認(返事)"]) | (chk["n_made_up"] > 0)]
    print(f"\n目で確かめるページ(失敗・返事の疑い・本文に無い行あり): {len(bad)}")
    if len(bad):
        print(bad[["election", "doc", "page", "status", "n_items", "n_made_up", "made_up_examples"]].to_string(index=False))
    log = ANSWERS / "calls.csv"
    if log.exists():
        print(f"\n料金の合計(本番のフォルダで呼んだ分): ${pd.read_csv(log)['cost_usd'].sum():.2f}")
    print(f"\n保存: {OUT}\n      {CHECK}")


if __name__ == "__main__":
    main()

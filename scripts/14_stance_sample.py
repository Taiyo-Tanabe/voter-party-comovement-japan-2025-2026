"""C 段階 ②: 立場のラベルの正解を作る行を、policies.parquet から乱数で選ぶ。

決まり(data/external/codebook.md の「集計の決まり」と同じ):
    - 目次のページ(data/external/segment_toc_pages.csv)の行は選ばない
    - 同じ党・同じ選挙の中で、文が全く同じ行の2回目以降は選ばない
選び方:
    - 党 × 選挙(21の組)ごとに選ぶ。行の多い党ばかりにならないように、各組の数は「行数の平方根」に比べて決める
    - 8つの争点の行が少なすぎないように、争点の言葉を含む行(当たり)と含まない行(そのほか)から半分ずつ選ぶ
      (言葉は当たりを付けるためだけに使う。ラベルは人が付ける)
    - 選んだ行を、組ごとに交互に 調整用(tune)と試験用(test)に分ける

出力:
    data/external/stance_gold/stance_gold_sheet.csv   ラベルを書き込む表(党・選挙は見せない。Excel で開ける UTF-8 BOM 付き)
    data/external/stance_gold/stance_gold_key.csv     id と、党・選挙・ページ・行の対応(書き込み中は開かない)

使い方: python scripts/14_stance_sample.py
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "interim" / "manifestos" / "policies.parquet"
TOC = ROOT / "data" / "external" / "segment_toc_pages.csv"
OUT = ROOT / "data" / "external" / "stance_gold"

N_TOTAL = 200
SEED = 0
# 8つの争点の当たりを付ける言葉(ラベルではない)
ISSUE_WORDS = ("防衛|安全保障|自衛隊|抑止|日米同盟|反撃能力|軍拡|軍事|核兵器|辺野古|米軍|インテリジェンス|スパイ"
               "|憲法|改憲|緊急事態条項"
               "|消費税"
               "|国債|財政|歳出|プライマリー"
               "|年金|社会保険|医療費|窓口負担|介護保険|生活保護"
               "|外国人|移民|難民|入管|育成就労|特定技能|永住|帰化"
               "|夫婦別姓|同性婚|旧姓|旧氏|LGBT|性的少数|ジェンダー"
               "|原発|原子力|再稼働")


def key_text(t):
    """同じ文かを比べる形: NFKC → 行頭の印・番号を消す → 空白を消す(13_segment_eval.py の norm と同じ考え方)"""
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"^[\s●○◯〇・･─━ー–—\-■□◆◇▶►※*❶-❿①-⑳]+", "", t)
    t = re.sub(r"^(\(\d{1,2}\)|\d{1,2}(?!\d)\.?)\s*", "", t)
    return re.sub(r"\s", "", t)


def allocate(sizes, n):
    """各組に選ぶ数: 行数の平方根に比べて n を分ける(組の行数を超えない)"""
    size = sizes.to_numpy()
    w = np.sqrt(size.astype(float))
    share = w / w.sum() * n
    alloc = np.minimum(np.floor(share), size).astype(int)
    rest = n - alloc.sum()
    order = np.argsort(-(share - alloc))                 # 端数の大きい組から1つずつ足す
    for i in np.tile(order, 10):
        if rest <= 0:
            break
        if alloc[i] < size[i]:
            alloc[i] += 1
            rest -= 1
    assert alloc.sum() == n, f"{n} 行を配り切れなかった(配れたのは {alloc.sum()} 行)"   # 黙って少なくならないように
    return pd.Series(alloc, index=sizes.index)


def main():
    p = pd.read_parquet(POLICIES)
    toc = pd.read_csv(TOC)
    toc_keys = set(zip(toc["election"], toc["doc"], toc["page"].astype(int)))
    p["in_toc"] = [(e, d, pg) in toc_keys for e, d, pg in zip(p["election"], p["doc"], p["page"])]
    p["key"] = p["text"].map(key_text)
    p["dup"] = p.duplicated(["election", "party", "key"], keep="first")
    pool = p[~p["in_toc"] & ~p["dup"] & (p["key"].str.len() > 0)].copy()
    pool["issue_hit"] = pool["text"].str.contains(ISSUE_WORDS)
    print(f"全 {len(p)} 行 → 目次 {int(p['in_toc'].sum())}・同じ文の2回目 {int(p['dup'].sum())} を除いて {len(pool)} 行")
    print(f"  争点の言葉を含む行: {int(pool['issue_hit'].sum())}")

    rng = np.random.default_rng(SEED)
    picked = []
    for hit in [True, False]:
        sub = pool[pool["issue_hit"] == hit]
        sizes = sub.groupby(["election", "party"]).size()
        alloc = allocate(sizes, N_TOTAL // 2)
        for (e, party), k in alloc.items():
            g = sub[(sub["election"] == e) & (sub["party"] == party)]
            picked.append(g.sample(k, random_state=int(rng.integers(1_000_000_000))))
    s = pd.concat(picked)

    # 組ごとに交互に tune / test(どちらにも同じような党・争点が入るように)
    # (組の中で混ぜてから組ごとに並べ、上から交互に付ける。行数が奇数の組の余りも、tune と test に交互に回る)
    # (2つ以上の列で並べるとき、pandas は同じ組の中の順番を変えない = 混ぜた順番が残る)
    s = s.sample(frac=1, random_state=SEED).sort_values(["election", "party", "issue_hit"])
    s["split"] = np.where(np.arange(len(s)) % 2 == 0, "tune", "test")
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)       # 表の並びも混ぜる(党ごとに固まらないように)
    s.insert(0, "id", [f"S{i + 1:03d}" for i in range(len(s))])

    print(f"\n選んだ行: {len(s)}(調整用 {int((s['split'] == 'tune').sum())}・試験用 {int((s['split'] == 'test').sum())})")
    print(s.groupby(["election", "party"]).size().unstack(0).fillna(0).astype(int).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    sheet = s[["id", "split", "heading", "text"]].copy()
    for c in ["kind", "topic", "stance", "unsure", "memo"]:
        sheet[c] = ""
    sheet.to_csv(OUT / "stance_gold_sheet.csv", index=False, encoding="utf-8-sig")
    s[["id", "split", "election", "party", "doc", "page", "item", "issue_hit"]].to_csv(
        OUT / "stance_gold_key.csv", index=False, encoding="utf-8")
    print(f"\n保存: {OUT / 'stance_gold_sheet.csv'}\n      {OUT / 'stance_gold_key.csv'}")


if __name__ == "__main__":
    main()

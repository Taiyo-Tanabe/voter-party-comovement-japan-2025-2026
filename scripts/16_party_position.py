"""D 段階 ①: Claude のラベル(15_stance_all.py の出力)を党ごとに集計して、党の位置と話題の量を出す。

決まり(data/external/codebook.md の「5. 集計の決まり」と、プロジェクトの「D 段階の設計」):
    - 主な集計と確かめの集計の2つを出す
        主な集計:   同じ党・同じ選挙の中で同じ文の行は最初の1行だけ / 目次のページは外す /
                    立場は 約束 + 説明 / 話題の量は 約束 だけ
        確かめの集計: 全部の行 / 目次のページも入れる / 立場は 約束 だけ / 話題の量は 約束 + 説明
    - 党の位置 = 党 × 選挙 × 争点ごとの立場(+1 / 0 / −1)の平均。行の数 n と 95% の幅も出す
    - 95% の幅はブートストラップで作る。同じページの行は似ているので、行ではなく**ページを選び直す**
    - 図と「動いた」の判断に使うのは n が MIN_N 行以上の組だけ(表には全部残す)
    - 両方の選挙にある党の「動いた」: 差の幅に加えて、比べる組の多さを Benjamini–Hochberg 法で補正する
    - 中道改革連合 2026 は、立憲 2025・公明 2025 のそれぞれと比べる(と、2党の平均を半分ずつ足した値とも)
    - 話題の量 = 約束の行の割合。前後の差は見せるだけで、「動いた」の判断はしない

使い方: python scripts/16_party_position.py

出力(reports/):
    party_position.csv        党 × 選挙 × 争点の平均・n・幅
    party_movement.csv        両方の選挙にある党の、争点ごとの動いた量・幅・補正した p 値
    chudo_compare.csv         中道 2026 と、立憲 2025・公明 2025 の差
    topic_share.csv           党 × 選挙 × 話題の、約束の行の割合
    topic_share_change.csv    両方の選挙にある党の、話題の割合の差と幅
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "processed" / "policies_stance.parquet"
TOC = ROOT / "data" / "external" / "segment_toc_pages.csv"
REPORTS = ROOT / "reports"

# 14_stance.py の争点と話題の一覧、14_stance_sample.py の「同じ文か」の比べ方をそのまま使う
def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s14 = _load("stance14", "14_stance.py")
sample14 = _load("sample14", "14_stance_sample.py")
ISSUES, TOPICS = s14.ISSUES, s14.TOPICS

BEFORE, AFTER = "2025_upper", "2026_lower"
MIN_N = 5            # 判断に使う最小の行数(目安。README で 3・10 にした場合も見る)
N_BOOT = 10_000      # ブートストラップの回数
ALPHA = 0.05
SEED = 0


def filter_rows(df, variant):
    """集計の決まりに従って行を絞る。返り値: (立場に使う行, 話題の量に使う行)"""
    if variant == "主な集計":
        d = df[~df.in_toc & ~df.dup]
        return d[d.llm_kind.isin(["約束", "説明"])], d[d.llm_kind == "約束"]
    if variant == "確かめの集計":
        return df[df.llm_kind == "約束"], df[df.llm_kind.isin(["約束", "説明"])]
    raise ValueError(f"知らない集計: {variant}")


def boot_means(g, rng):
    """ページを選び直して平均を N_BOOT 回作る。g: 1つの組の行(stance_num と page_key がある)"""
    per_page = g.groupby("page_key").stance_num.agg(["sum", "count"])
    sums, counts = per_page["sum"].to_numpy(), per_page["count"].to_numpy()
    idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))       # 各回、ページの数だけページを引く(戻して引く)
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return round(lo, 3), round(hi, 3)


def boot_p(diffs):
    """差の両側 p 値(ブートストラップの差が 0 の向こう側に出た割合 × 2)"""
    return min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


def bh(p):
    """Benjamini–Hochberg 法で補正した p 値"""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]                   # 大きい順に、それより後ろの最小値をとる
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def positions(st, variant, rng):
    """党 × 選挙 × 争点の平均・n・幅。ブートストラップの結果も返す(差を作るのに使う)"""
    rows, boots = [], {}
    for (e, party, topic), g in st.groupby(["election", "party", "llm_topic"]):
        b = boot_means(g, rng)
        boots[(e, party, topic)] = b
        lo, hi = ci(b)
        rows.append({"集計": variant, "election": e, "party": party, "topic": topic,
                     "n": len(g), "n_pages": g.page_key.nunique(), "mean": round(g.stance_num.mean(), 3),
                     "ci_lo": lo, "ci_hi": hi, "行が少ない": len(g) < MIN_N, "幅が0": lo == hi})
    return pd.DataFrame(rows), boots


def movement(pos, boots, variant):
    """両方の選挙にある党の、争点ごとの動いた量(後 − 前)"""
    both = sorted(set(pos[pos.election == BEFORE].party) & set(pos[pos.election == AFTER].party))
    rows = []
    for party in both:
        for topic in ISSUES:
            a, b = boots.get((BEFORE, party, topic)), boots.get((AFTER, party, topic))
            if a is None or b is None:
                continue
            pos_前 = pos[(pos.election == BEFORE) & (pos.party == party) & (pos.topic == topic)].iloc[0]
            pos_後 = pos[(pos.election == AFTER) & (pos.party == party) & (pos.topic == topic)].iloc[0]
            d = b - a                                                 # 前と後は別々に選び直したので、そのまま引ける
            lo, hi = ci(d)
            rows.append({"集計": variant, "party": party, "topic": topic, "n_前": pos_前.n, "n_後": pos_後.n,
                         "mean_前": pos_前["mean"], "mean_後": pos_後["mean"],
                         "動いた量": round(pos_後["mean"] - pos_前["mean"], 3),
                         "ci_lo": lo, "ci_hi": hi, "p": round(boot_p(d), 4),
                         "判断に使う": min(pos_前.n, pos_後.n) >= MIN_N})
    m = pd.DataFrame(rows)
    m["p_補正"] = np.nan
    use = m["判断に使う"]
    if use.any():
        m.loc[use, "p_補正"] = bh(m.loc[use, "p"]).round(4)          # 補正は、判断に使う組の中だけで
    m["動いた"] = use & (m["p_補正"] < ALPHA)
    return m


def chudo(pos, boots, variant):
    """中道 2026 と、立憲 2025・公明 2025・2党の平均(半分ずつ)の差"""
    rows = []
    for topic in ISSUES:
        c = boots.get((AFTER, "中道", topic))
        r, k = boots.get((BEFORE, "立憲", topic)), boots.get((BEFORE, "公明", topic))
        if c is None:
            continue

        def get(e, p):
            return pos[(pos.election == e) & (pos.party == p) & (pos.topic == topic)]

        row = {"集計": variant, "topic": topic, "n_中道": int(get(AFTER, "中道").n.iloc[0]),
               "mean_中道": get(AFTER, "中道")["mean"].iloc[0]}
        for name, b in [("立憲", r), ("公明", k)]:
            if b is None:
                continue
            pos_相手 = get(BEFORE, name)
            lo, hi = ci(c - b)
            row.update({f"n_{name}": int(pos_相手.n.iloc[0]), f"mean_{name}": pos_相手["mean"].iloc[0],
                        f"中道−{name}": round(row["mean_中道"] - pos_相手["mean"].iloc[0], 3),
                        f"中道−{name}_lo": lo, f"中道−{name}_hi": hi})
        if r is not None and k is not None:
            half = (r + k) / 2
            lo, hi = ci(c - half)
            row.update({"中道−2党の平均": round(row["mean_中道"] - (row["mean_立憲"] + row["mean_公明"]) / 2, 3),
                        "中道−2党の平均_lo": lo, "中道−2党の平均_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def topic_share(tp, variant, rng):
    """党 × 選挙 × 話題の約束の行の割合と、両方の選挙にある党の差(ページを選び直した幅つき)"""
    all_topics = ISSUES + TOPICS
    shares, boots = [], {}
    for (e, party), g in tp.groupby(["election", "party"]):
        mat = pd.crosstab(g.page_key, g.llm_topic).reindex(columns=all_topics, fill_value=0)   # ページ × 話題の行の数
        m = mat.to_numpy()
        idx = rng.integers(0, len(m), size=(N_BOOT, len(m)))
        tot = m[idx].sum(axis=(1, 2))
        boots[(e, party)] = m[idx].sum(axis=1) / tot[:, None]          # 各回の、話題ごとの割合
        cnt = m.sum(axis=0)
        for j, t in enumerate(all_topics):
            shares.append({"集計": variant, "election": e, "party": party, "topic": t,
                           "行": int(cnt[j]), "約束の行の全部": int(m.sum()), "割合": round(cnt[j] / m.sum(), 4)})
    shares = pd.DataFrame(shares)
    both = sorted({p for e, p in boots if e == BEFORE} & {p for e, p in boots if e == AFTER})
    rows = []
    for party in both:
        d = boots[(AFTER, party)] - boots[(BEFORE, party)]
        for j, t in enumerate(all_topics):
            a = shares[(shares.election == BEFORE) & (shares.party == party) & (shares.topic == t)].iloc[0]
            b = shares[(shares.election == AFTER) & (shares.party == party) & (shares.topic == t)].iloc[0]
            lo, hi = ci(d[:, j])
            rows.append({"集計": variant, "party": party, "topic": t, "割合_前": a["割合"], "割合_後": b["割合"],
                         "差": round(b["割合"] - a["割合"], 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    return shares, pd.DataFrame(rows)


def main():
    df = pd.read_parquet(LABELS)
    toc = pd.read_csv(TOC)
    toc_keys = set(zip(toc["election"], toc["doc"], toc["page"].astype(int)))
    df["in_toc"] = [(e, d, p) in toc_keys for e, d, p in zip(df.election, df.doc, df.page)]
    df["dup"] = df.assign(key=df.text.map(sample14.key_text)).duplicated(["election", "party", "key"], keep="first")
    df["page_key"] = df.doc + "_" + df.page.astype(str)

    # 争点の行で、立場が読めない答え(決まりに合わない答え)は、平均に入れられないので外す
    is_issue = df.llm_topic.isin(ISSUES)
    bad = is_issue & ~df.llm_stance.isin(["1", "-1", "0"])
    print(f"全 {len(df)} 行 / 目次のページ {int(df.in_toc.sum())} / 同じ文の2回目以降 {int(df.dup.sum())} / "
          f"争点なのに立場が無い行 {int(bad.sum())}(外す)")
    df["stance_num"] = pd.to_numeric(df.llm_stance.where(is_issue & ~bad), errors="coerce")

    rng = np.random.default_rng(SEED)
    out = {k: [] for k in ["position", "movement", "chudo", "share", "share_change"]}
    for variant in ["主な集計", "確かめの集計"]:
        st, tp = filter_rows(df, variant)
        st = st[st.stance_num.notna()]
        pos, boots = positions(st, variant, rng)
        mov = movement(pos, boots, variant)
        sh, shc = topic_share(tp, variant, rng)
        out["position"].append(pos)
        out["movement"].append(mov)
        out["chudo"].append(chudo(pos, boots, variant))
        out["share"].append(sh)
        out["share_change"].append(shc)
        used = mov[mov["判断に使う"]]
        print(f"\n[{variant}] 立場の行 {len(st)} / 話題の量の行 {len(tp)}")
        print(f"  党 × 選挙 × 争点 {len(pos)} 組(うち {MIN_N} 行以上 {int((~pos['行が少ない']).sum())})")
        print(f"  前後を比べた組 {len(mov)}(判断に使う {len(used)})→ 動いた {int(mov['動いた'].sum())} 組"
              f"(補正しないと {int((used.p < ALPHA).sum())} 組)")
        for k in [3, 10]:                                              # 線引きを変えた確かめ
            u = mov[np.minimum(mov.n_前, mov.n_後) >= k]
            n_moved = int((bh(u.p) < ALPHA).sum()) if len(u) else 0
            print(f"    線引きを {k} 行にすると: 判断に使う {len(u)} 組 → 動いた {n_moved} 組")
        moved = mov[mov["動いた"]].sort_values("p_補正")
        if len(moved):
            print(moved[["party", "topic", "n_前", "n_後", "mean_前", "mean_後", "動いた量", "ci_lo", "ci_hi", "p_補正"]]
                  .to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    names = {"position": "party_position", "movement": "party_movement", "chudo": "chudo_compare",
             "share": "topic_share", "share_change": "topic_share_change"}
    for k, name in names.items():
        pd.concat(out[k]).to_csv(REPORTS / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n保存: reports/ の {', '.join(n + '.csv' for n in names.values())}")


if __name__ == "__main__":
    sys.exit(main())

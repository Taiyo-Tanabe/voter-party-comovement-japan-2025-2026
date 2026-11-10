"""C 段階 ④: Claude に、正解の行と同じ「見出し + 文」を渡して立場のラベルを付けさせ、本人の正解と比べる。

決まり:
    - Claude に渡すのは、本人に見せたのと同じ見出しと文だけ(党・選挙は渡さない)
    - 決まりは data/external/codebook.md をそのまま前置き(system)に入れる。前置きはキャッシュして安くする
    - 20行ずつまとめて渡し、JSON で答えさせる
    - 比べるのは調整用(tune)の100行だけ。試験用(test)は、プロンプトの直しが全部終わってから
      --split test --final を付けて1回だけ使う

使い方(プロジェクトのフォルダで):
    python scripts/14_stance.py                  # 料金の見積もりを出すだけ(Claude は呼ばない)
    python scripts/14_stance.py --run            # 調整用100行を判定して、一致度(κ)を出す
    python scripts/14_stance.py --run --tag v2   # プロンプトを直したら tag を変える(答えが混ざらないように)

出力(tag ごと):
    data/interim/manifestos/stance/gold_<tag>/batch_XXX.txt   Claude の答え(保存済みの組は呼ばない。途中で止めても続きから)
    data/interim/manifestos/stance/gold_<tag>/calls.csv       料金の記録
    reports/stance_gold_<tag>_<split>.csv                     1行ずつの比べ(正解と Claude の答え)
    reports/stance_kappa.csv                                  一致度の記録(実行のたびに足す)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import PRICES, call_claude, make_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = ROOT / "data" / "external" / "codebook.md"
SHEET = ROOT / "data" / "external" / "stance_gold" / "stance_gold_sheet.csv"
STANCE_DIR = ROOT / "data" / "interim" / "manifestos" / "stance"
REPORTS = ROOT / "reports"
KAPPA_LOG = REPORTS / "stance_kappa.csv"

MODEL = "claude-sonnet-5"
EFFORT = "medium"           # 13_segment と同じ(本人が決定)
BATCH = 20                  # 1回に渡す行数

ISSUES = ["安全保障", "憲法", "消費税", "財政", "社会保障", "外国人", "ジェンダー", "エネルギー"]   # A: 立場を付ける
TOPICS = ["子育て・教育", "労働・賃金", "経済・産業", "農林水産", "地方・行政", "政治改革", "外交", "環境",
          "防災", "デジタル", "医療・福祉の現場", "住まい", "治安・司法", "人権・暮らし", "その他"]       # B: 話題だけ
KINDS = ["約束", "説明", "政策でない"]

SYSTEM_HEAD = """あなたは日本の政党の選挙公約を分類する作業者です。
次の「コードブック」の決まりだけに従って、渡された行それぞれにラベルを付けてください。
党の名前から推測せず、書かれている文と見出しだけで判定してください。

===== コードブック =====
"""

PROMPT_HEAD = """次の各行に、kind・topic・stance を付けてください。

答えは JSON の配列だけを書き、説明の文は書かないでください。形:
[{"id": "S001", "kind": "約束", "topic": "消費税", "stance": 1}, ...]

- kind: "約束" / "説明" / "政策でない"
- topic: コードブックの A の8つか B の15のどれか1つ。kind が "政策でない" のときは ""
- stance: topic が A の8つのときだけ 1 / -1 / 0。それ以外は null

===== 行 =====
"""


def build_prompt(rows):
    lines = []
    for r in rows.itertuples():
        head = r.heading if r.heading else "(なし)"
        lines.append(f"id: {r.id}\n見出し: {head}\n文: {r.text}\n")
    return PROMPT_HEAD + "\n".join(lines)


def parse_answer(text, ids):
    """Claude の答え(JSON)を読み、決まりに合わない値を印にする。返り値: id ごとの辞書"""
    m = re.search(r"\[.*\]", text, re.S)                     # 前後に文が付いても配列だけ取り出す
    if not m:
        raise ValueError("JSON の配列が見つからない")
    out = {}
    for d in json.loads(m.group(0)):
        i = str(d.get("id", ""))
        kind = str(d.get("kind") or "")
        topic = str(d.get("topic") or "")
        st = d.get("stance")
        stance = "" if st is None or st == "" else str(int(st))   # 1 / -1 / 0 を文字にそろえる(正解の表と同じ形)
        bad = []
        if kind not in KINDS:
            bad.append("kind")
        if kind == "政策でない" and topic:
            bad.append("政策でないのに topic")
        if kind != "政策でない" and topic not in ISSUES + TOPICS:
            bad.append("topic")
        if topic in ISSUES and stance not in ("1", "-1", "0"):
            bad.append("stance なし")
        if topic not in ISSUES and stance:
            bad.append("話題なのに stance")
        out[i] = {"llm_kind": kind, "llm_topic": topic, "llm_stance": stance, "llm_bad": "・".join(bad)}
    missing = [i for i in ids if i not in out]
    if missing:
        raise ValueError(f"答えに無い id: {missing}")
    return out


def kappa(a, b):
    """2つの列の一致度。行が無い・全部同じ値のときは計算できないので None"""
    if len(a) == 0 or (len(set(a)) == 1 and len(set(b)) == 1 and set(a) == set(b)):
        return None
    return round(cohen_kappa_score(a, b), 3)


def score(df):
    """種類・争点・立場の一致度と一致率。迷った行を入れた場合と除いた場合の両方"""
    res = []
    for name, d in [("全部", df), ("迷った行を除く", df[df.unsure != "1"])]:
        row = {"範囲": name, "行数": len(d)}
        row["kind_一致率"] = round((d.kind == d.llm_kind).mean(), 3)
        row["kind_κ"] = kappa(d.kind, d.llm_kind)
        pol = d[d.kind != "政策でない"]                       # 正解で政策の行だけ、争点・話題を比べる
        row["topic_一致率"] = round((pol.topic == pol.llm_topic).mean(), 3)
        row["topic_κ"] = kappa(pol.topic, pol.llm_topic)
        a = d[d.topic.isin(ISSUES)]                           # 正解が A の争点の行だけ、立場を比べる
        llm = [s if t == g else "別の争点" for s, t, g in zip(a.llm_stance, a.llm_topic, a.topic)]
        row["stance_行数"] = len(a)
        row["stance_一致率"] = round((a.stance == pd.Series(llm, index=a.index)).mean(), 3) if len(a) else None
        row["stance_κ"] = kappa(list(a.stance), llm)          # 争点が違えば「別の争点」として外れに数える
        same = a[a.llm_topic == a.topic]                      # 争点が合った行だけで立場を比べる(向きの読み方だけの一致度)
        row["stance_同じ争点の行数"] = len(same)
        row["stance_同じ争点のκ"] = kappa(same.stance, same.llm_stance)
        res.append(row)
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Claude を呼ぶ(付けないと見積もりだけ)")
    ap.add_argument("--tag", default="v1", help="プロンプトの版。直したら変える")
    ap.add_argument("--split", default="tune", choices=["tune", "test"])
    ap.add_argument("--final", action="store_true", help="試験用を使うときに必要(1回だけ)")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("試験用は、プロンプトの直しが全部終わってから --final を付けて1回だけ使う")

    sheet = pd.read_csv(SHEET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = sheet[sheet.split == args.split].sort_values("id").reset_index(drop=True)
    system = SYSTEM_HEAD + CODEBOOK.read_text(encoding="utf-8")
    batches = [rows.iloc[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    out_dir = STANCE_DIR / f"gold_{args.tag}"

    # 料金の見積もり(日本語は1字 ≒ 1トークンとして大まかに。考える分は1回 3,000 トークンと見る)
    p_in, p_out = PRICES[MODEL]
    sys_tok = len(system)
    row_tok = sum(len(build_prompt(b)) for b in batches)
    est = (sys_tok * p_in * 1.25 + sys_tok * p_in * 0.1 * (len(batches) - 1)
           + row_tok * p_in + len(batches) * (3000 + 60 * BATCH) * p_out) / 1e6
    print(f"{args.split} {len(rows)}行・{len(batches)}回・見積もり 約 ${est:.2f}(tag {args.tag})")
    if not args.run:
        print("Claude を呼ぶときは --run を付ける")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client()
    calls_path = out_dir / "calls.csv"
    answers = {}
    for k, b in enumerate(batches):
        path = out_dir / f"{args.split}_batch_{k:03d}.txt"
        if not (path.exists() and path.read_text(encoding="utf-8").strip()):
            text, rec = call_claude(client, MODEL, build_prompt(b), effort=EFFORT, system=system, cache=True)
            path.write_text(text, encoding="utf-8")
            rec.update({"split": args.split, "batch": k, "tag": args.tag})
            pd.DataFrame([rec]).to_csv(calls_path, mode="a", index=False, header=not calls_path.exists(),
                                       encoding="utf-8-sig")
            print(f"  {k + 1}/{len(batches)} ${rec['cost_usd']} {rec['seconds']}秒 {rec['warning']}")
        answers.update(parse_answer(path.read_text(encoding="utf-8"), list(b.id)))

    df = rows.join(pd.DataFrame.from_dict(answers, orient="index"), on="id")
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / f"stance_gold_{args.tag}_{args.split}.csv", index=False, encoding="utf-8-sig")
    bad = df[df.llm_bad != ""]
    if len(bad):
        print(f"決まりに合わない答え {len(bad)}行: {list(bad.id)}")

    s = score(df)
    s.insert(0, "split", args.split)
    s.insert(0, "tag", args.tag)
    print(s.to_string(index=False))
    s.to_csv(KAPPA_LOG, mode="a", index=False, header=not KAPPA_LOG.exists(), encoding="utf-8-sig")
    if calls_path.exists():
        c = pd.read_csv(calls_path, encoding="utf-8-sig")
        c = c[(c.tag == args.tag) & (c.split == args.split)]
        print(f"料金の合計 ${c.cost_usd.sum():.3f}")


if __name__ == "__main__":
    main()

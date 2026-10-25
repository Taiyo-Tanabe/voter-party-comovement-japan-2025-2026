"""13_segment の方法比べ: ページの本文を「1行 = 1政策」に分ける方法を、手で作った正解と比べて採点する。

正解: data/external/segment_gold/<選挙>_<文書>_p<ページ>.txt(書き方は同じフォルダの _書き方.md)
    - 1行 = 1政策。`#` で始まる行は見出し(次の `#` まで、下の政策に付く)
    - 行末に ` ??` の行は「決まりで決められない行」→ 採点から外す
    - 行頭の印(● ○ ・ ─ ❶ ① など)と空白は、比べる前に取り除く

使い方(プロジェクトのフォルダで):
    python scripts/13_segment_eval.py                          # 何もしない場合(lines)とルール版(rule)を採点(無料)
    python scripts/13_segment_eval.py --methods lines rule claude   # Claude 版も(API の料金がかかる。答えは保存して使い回す)

出力:
    data/interim/manifestos/segment_eval/claude/<選挙>_<文書>_p<ページ>.txt と calls.csv(Claude の答えと呼んだ記録)
    画面: ページごと・方法ごとの 再現率・適合率・F1・作文の数・見出しの一致率
    reports/segment_eval.csv
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "interim" / "manifestos" / "pages_ocr.parquet"
GOLD_DIR = ROOT / "data" / "external" / "segment_gold"
REPORT = ROOT / "reports" / "segment_eval.csv"

GOLD_NAME = re.compile(r"^(\d{4}_\w+?)_(\w+)_p(\d{3})\.txt$")   # 2025_upper_ldp_p005.txt → 選挙・文書・ページ
UNSURE = "??"
# 行頭の印。数字は本文の一部のこともある(「1990年度には…」「10 年後の主力産業…」)ので、政策の行では消さない
MARKS = "●○◯・･─━ー–—-■□◆◇▶►※*❶❷❸❹❺❻❼❽❾❿①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
LEAD_MARKS = re.compile(f"^[\\s{re.escape(MARKS)}]+")
# 見出しの行だけは、頭の番号も消す(「01 経済財政関連施策」「(2) 米の価格高騰…」「3 税金は…」)
LEAD_NUMBER = re.compile(r"^(\(\d{1,2}\)|\d{1,2}(?!\d)[\.．]?)\s*")   # (?!\d): 西暦など3桁以上の数字は番号とみなさない

def norm(s):
    """比べるための形: NFKC → 行頭の印を消す → 空白をすべて消す"""
    s = unicodedata.normalize("NFKC", s)
    s = LEAD_MARKS.sub("", s)
    return re.sub(r"\s", "", s)


def norm_heading(s):
    """見出しを比べるための形: 印と頭の番号を消す"""
    s = unicodedata.normalize("NFKC", s).strip()
    s = LEAD_MARKS.sub("", s)
    s = LEAD_NUMBER.sub("", s)
    s = LEAD_MARKS.sub("", s)                       # 「01 ●…」のように番号の後に印が続くこともある
    return re.sub(r"\s", "", s)


def parse_items(text):
    """「1行 = 1政策、# は見出し」の形の文字列を、政策のリストにする(正解と、各方法の出力の両方に使う)。
    返り値: [{"text": 比べる形, "raw": 元の行, "heading": 見出し(比べる形 or ""),
              "unsure": ?? の行か, "heading_unsure": 付いている見出しが ?? か}]"""
    items, heading, heading_unsure = [], "", False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        unsure = line.endswith(UNSURE)
        if unsure:
            line = line[: -len(UNSURE)].rstrip()
        if line.startswith("#"):
            heading, heading_unsure = norm_heading(line.lstrip("#")), unsure
            continue
        t = norm(line)
        if t:
            items.append({"text": t, "raw": line, "heading": heading, "unsure": unsure,
                          "heading_unsure": heading_unsure})
    return items


def load_gold():
    """正解のファイルを全部読む: {(選挙, 文書, ページ): 政策のリスト}"""
    gold = {}
    for f in sorted(GOLD_DIR.glob("*.txt")):
        m = GOLD_NAME.match(f.name)
        if not m:
            continue
        gold[(m[1], m[2], int(m[3]))] = parse_items(f.read_text(encoding="utf-8"))
    return gold


def score(gold_items, pred_items, page_text):
    """1ページ分の採点。
    一致 = 比べる形の文字列が同じ(同じ文字列が2回あれば2回まで数える)。
    ?? の正解の行は数えない。方法がその行を出していても、外れには数えない。"""
    sure = Counter(g["text"] for g in gold_items if not g["unsure"])
    unsure = {g["text"] for g in gold_items if g["unsure"]}
    preds = [p for p in pred_items if p["text"] not in unsure]
    hit = sum((sure & Counter(p["text"] for p in preds)).values())
    n_gold, n_pred = sum(sure.values()), len(preds)
    rec = hit / n_gold if n_gold else 0.0
    prec = hit / n_pred if n_pred else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    # 作文: 元の本文(改行・空白を除いてつないだもの)の中に、そのまま続けて現れない政策の行。
    # 言い換え・書き足し・語の抜け(「最低20日に」を落とす など)が見つかる。
    # 離れた行どうしをつないだもの(間の行を飛ばした)も数えられるので、数が出たら中身を目で確かめる
    flat = re.sub(r"\s", "", unicodedata.normalize("NFKC", page_text))
    made_up = [p["raw"] for p in preds if p["text"] not in flat]
    # 見出しの一致: 一致した政策のうち、付いている見出しも同じ割合。
    # 正解の見出しが ?? (見出しかどうか決めていない)の政策は、ここでは数えない
    gold_head = {}
    for g in gold_items:
        gold_head.setdefault(g["text"], g)
    matched = [p for p in preds if p["text"] in sure and not gold_head[p["text"]]["heading_unsure"]]
    head_ok = sum(p["heading"] == gold_head[p["text"]]["heading"] for p in matched)
    return {"n_gold": n_gold, "n_pred": n_pred, "hit": hit,
            "recall": rec, "precision": prec, "f1": f1,
            "made_up": len(made_up), "made_up_examples": " / ".join(m[:30] for m in made_up[:3]),
            "head_n": len(matched), "head_ok": head_ok,
            "heading_acc": head_ok / len(matched) if matched else float("nan")}


# ---------- 方法 ----------
def seg_lines(page_text, doc):
    """比べる基準: 元の本文の1行をそのまま1政策とみなす(何もしない場合の点数)"""
    return parse_items(page_text.replace("#", "＃"))   # 本文の # は見出しにしない(全角に逃がす)


# ---- ルール版 ----
# 党ごとに違うのは「どの印が見出しか」だけにする(それ以外の決まりは全党共通。6ページに合わせすぎないため)
HEAD_MARKS = {
    "jcp": "●",                 # 共産: ● は小見出し、─ が政策
    "reiwa": "❶❷❸❹❺❻❼❽❾❿",      # れいわ: ❷ などは柱の番号(見出し)
}
ITEM_MARKS = "●○◯・─━■◆◇❶❷❸❹❺❻❼❽❾❿①②③④⑤⑥⑦⑧⑨⑩"
HEAD_PATTERN = re.compile(r"^(\d{2}\s|\(\d{1,2}\)|〈.+〉$)")       # 「01 経済…」「(2) 米の…」「〈物価高対策〉」
NOISE = re.compile(r"^(https?://|etc\.?$|Column$|Step\s*\d+$|具体的にやること$|もっと詳しく知りたい)")
SENT_END = "。?!"


def seg_rule(page_text, doc):
    """印で区切るルール版。
    - 行の頭の印で新しい政策を始める(党ごとに決めた見出しの印・番号の形・〈 〉は見出し)
    - 印のない行: 見出しの途中なら見出しにつなぐ / 政策の後なら同じ政策につなぐ(折れた行・2文目)/
      それ以外は文章として集め、最後に「。」「?」「!」で1文ずつに分ける
    - 印のない短い行で、次の行が印で始まるものは見出しとみなす
    - URL・etc.・「具体的にやること」などの案内は消す
    並び順の崩れは直せない(見出しの付き方に影響する)"""
    head_marks = HEAD_MARKS.get(doc, "")
    lines = []
    for raw in page_text.splitlines():
        raw = unicodedata.normalize("NFKC", raw).strip()
        lines += [x.strip() for x in re.split(r"\s+(?=[●○])", raw) if x.strip()]   # 「●憲法9条の改正 ●旧姓使用」→ 2行
    lines = [l for l in lines if not NOISE.match(l)]

    def kind(l):
        if l[0] in head_marks or HEAD_PATTERN.match(l):
            return "head"
        if l[0] in ITEM_MARKS:
            return "item"
        return None

    blocks = []                                         # [種類, 文字列] のリスト
    for i, l in enumerate(lines):
        k = kind(l)
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        cur = blocks[-1] if blocks else None
        if k:
            blocks.append([k, l])
        elif cur and cur[0] == "head" and cur[1][-1] not in SENT_END and not l.endswith("。"):
            cur[1] += l                                 # 折れた見出しの続き
        elif cur and cur[0] == "item" and cur[1][-1] not in SENT_END:
            cur[1] += l                                 # 政策の途中で折れた行(文がまだ終わっていない)
        elif len(l) <= 45 and not l.endswith("。") and (nxt is None or kind(nxt)):
            blocks.append(["head", l])                  # 印のない短い行の次が印 → 見出し
        elif cur and cur[0] == "item":
            cur[1] += l                                 # 政策の2文目・かっこ書きの続き
        elif cur and cur[0] == "prose" and cur[1][-1] not in SENT_END:
            cur[1] += l                                 # 文章の途中
        else:
            blocks.append(["prose", l])

    out = []
    for k, t in blocks:
        if k == "head":
            out.append("#" + t)
        elif k == "item":
            out.append(t)
        else:                                           # 文章は1文ずつ
            out += [x for x in re.split(r"(?<=[。?!])", t) if x.strip()]
    return parse_items("\n".join(out))


# ---- Claude 版 ----
CLAUDE_MODEL = "claude-sonnet-5"
CLAUDE_DIR = ROOT / "data" / "interim" / "manifestos" / "segment_eval" / "claude"   # 答えを保存して使い回す(二重に払わない)
PROMPT = """次の文章は、日本の政党の選挙公約(マニフェスト)の1ページから取り出した文字です。
PDF から機械的に取り出したので、紙の行の幅で文が途中で改行され、見出しや項目の順番が崩れていることがあります。
これを「1行 = 1つの政策」の形に書き直してください。

決まり:
- ●・○・・・─・❶・① などの印1つ分を1行にする(文が2つ以上あっても1行)。1行に印が2つあれば2行に分ける
- 印のない文章は、「。」「?」「!」までの1文を1行にする
- 途中で改行されている文は、つないで1行にする
- 見出し(政策そのものではなく、まとまりの名前)は、行の頭に # を付けて1行にする。見出しは、その下の政策に付く
- 見出しや項目の順番が崩れていると思われるときは、元の紙面の順番(見出し → その中身)に並べ直す
- ページ番号・ヘッダー・フッター・URL・etc.・「詳しくはこちら」のような案内は消す
- 文の中身は一字も変えない。言い換え・要約・書き足しは一切しない。行の頭の印は消してよい
- 書き直した本文だけを出力する(説明や前置き、マークダウンの囲みは付けない)

文章:
{text}"""


_client = None


def seg_claude(page_text, doc, key=None, force=False, effort=None, base=CLAUDE_DIR):
    """Claude に「1行 = 1政策」に書き直させる。答えは保存し、次からはそれを使う(--force で読み直し)。
    effort を指定したときは、答えを別のフォルダ(claude_low など)に保存する(設定ごとに比べるため)。
    base: 答えを保存するフォルダ(13_segment.py の本番では別のフォルダを渡す)"""
    global _client
    folder = base if effort is None else base.with_name(f"{base.name}_{effort}")
    out = folder / f"{key[0]}_{key[1]}_p{key[2]:03d}.txt"
    if out.exists() and not force:
        saved = out.read_text(encoding="utf-8")
        if saved.strip():                               # 空のファイル(前に失敗した答え)は使わず、呼び直す
            return parse_items(saved)
    from llm_common import call_claude, make_client
    if _client is None:
        _client = make_client()
    text, meta = call_claude(_client, CLAUDE_MODEL, PROMPT.format(text=page_text), effort=effort)
    if meta["warning"]:
        print(f"  [注意] {out.name}: {meta['warning']}")
    print(f"  Claude{'' if effort is None else '(' + effort + ')'} {out.name}: 出力 {meta['output_tokens']}(うち考えた {meta['thinking_tokens']})トークン"
          f" ${meta['cost_usd']:.3f} {meta['seconds']}秒 stop_reason={meta['stop_reason']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not meta["warning"]:                             # 空・打ち切りの答えは保存しない(次の実行で呼び直す)
        out.write_text(text, encoding="utf-8")
    log = folder / "calls.csv"                          # 呼んだ記録(料金・トークン・警告)
    row = pd.DataFrame([{"file": out.name, **meta}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False, encoding="utf-8")
    return parse_items(text)


METHODS = {"lines": seg_lines, "rule": seg_rule, "claude": seg_claude,
           "claude-medium": seg_claude, "claude-low": seg_claude}   # claude-○○ = effort を下げた Claude(考える量を減らして安くする)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lines", "rule"], choices=list(METHODS),
                    help="claude は API の料金がかかるので、指定したときだけ動く")
    ap.add_argument("--force", action="store_true", help="Claude の保存した答えを使わず、読み直す")
    args = ap.parse_args()

    pages = pd.read_parquet(PAGES)
    pages = pages[pages["kind"] == "main"].set_index(["election", "doc", "page"])
    gold = load_gold()

    # 正解の点検: 正解の行が、元の本文にそのまま含まれているか(一字でも書き換えていたら見つかる)
    for key, items in gold.items():
        bad = score(items, items, pages.loc[key, "text"])
        if bad["made_up"]:
            print(f"[正解の点検] {key}: 元の本文に無い行 {bad['made_up']} → {bad['made_up_examples']}")

    rows = []
    for name in args.methods:
        for key, items in gold.items():
            text = pages.loc[key, "text"]
            if name.startswith("claude"):
                effort = name.split("-")[1] if "-" in name else None       # "claude-low" → "low"、"claude" → None(初期値 high)
                pred = seg_claude(text, key[1], key, args.force, effort)
            else:
                pred = METHODS[name](text, key[1])
            r = score(items, pred, text)
            rows.append({"method": name, "election": key[0], "doc": key[1], "page": key[2], **r})
    res = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORT, index=False, encoding="utf-8")

    cols = ["method", "doc", "page", "n_gold", "n_pred", "hit", "recall", "precision", "f1", "made_up", "heading_acc"]
    print(res[cols].round(2).to_string(index=False))
    tot = res.groupby("method")[["n_gold", "n_pred", "hit", "made_up", "head_n", "head_ok"]].sum()
    tot["recall"] = tot["hit"] / tot["n_gold"]
    tot["precision"] = tot["hit"] / tot["n_pred"]
    tot["f1"] = 2 * tot["recall"] * tot["precision"] / (tot["recall"] + tot["precision"])
    tot["heading_acc"] = tot["head_ok"] / tot["head_n"]      # 見出しの一致も、数を足してから割り直す
    print("\n全体(6ページ合計)")
    print(tot.round(3).to_string())
    print(f"\n保存: {REPORT}")


if __name__ == "__main__":
    main()

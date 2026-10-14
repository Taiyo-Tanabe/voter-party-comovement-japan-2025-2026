"""OCR の方法を比べる: 同じ切り出し画像を Tesseract と Claude(API)で読ませ、手で作った正解との文字の間違い率(CER)を出す。

使い方(プロジェクトのフォルダで):
    python scripts/12_ocr_eval.py                     # まだ結果のない方法・画像だけ実行して採点
    python scripts/12_ocr_eval.py --methods tesseract # Tesseract だけ
    python scripts/12_ocr_eval.py --force             # 全部やり直す(API の料金がもう一度かかる)

入力:
    data/external/ocr_eval_pages.csv   どの文書の何ページのどこを切り出すか(px 座標は dpi で描いたときの値)
    data/external/ocr_gt/<id>.txt      手で打ち込んだ正解
    .env                               ANTHROPIC_API_KEY=...(Claude を使うとき)
出力:
    data/interim/ocr_eval/crops/<id>.png         切り出し画像
    data/interim/ocr_eval/out/<method>/<id>.txt  各方法の読み取り結果(あれば再利用 = 料金を二重に払わない)
    reports/ocr_eval.csv                          id × 方法 ごとの CER など

採点の前に、正解・結果の両方を NFKC で全角半角をそろえ、空白・改行をすべて消す
(改行の位置や空白の入れ方の違いは間違いとして数えない)。
"""
import argparse
import base64
import io
import os
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "external" / "ocr_eval_pages.csv"
GT_DIR = ROOT / "data" / "external" / "ocr_gt"
PDF_DIR = ROOT / "data" / "raw" / "manifestos"
WORK = ROOT / "data" / "interim" / "ocr_eval"
REPORT = ROOT / "reports" / "ocr_eval.csv"

# 方法ごとの設定。Claude は長い辺がこれを超えると API 側で縮小されるので、先に自分で縮めて条件をそろえる
# (公式ドキュメント Vision: Claude 4.7 以降は 2576px、それ以外は 1568px)
METHODS = {
    "tesseract":     {"kind": "tesseract"},
    "claude-haiku":  {"kind": "claude", "model": "claude-haiku-4-5-20251001", "max_edge": 1568,
                      "price_in": 1.0, "price_out": 5.0},     # USD / 100万トークン(2026-10 の料金表)
    "claude-sonnet": {"kind": "claude", "model": "claude-sonnet-5", "max_edge": 2576,
                      "price_in": 2.0, "price_out": 10.0},
}

PROMPT = """この画像は日本の政党の選挙公約の一部です。画像の中の文字をすべて、そのまま書き起こしてください。
- 上から下、左から右の順に書く。枠や囲みが並んでいるときは、枠ごとに左上→右上→左下→右下の順
- 見出しも含める。イラストの中の文字やロゴは除く
- 要約・言い換え・補足・説明は一切しない。読めない文字は 〓 にする
- 書き起こした本文だけを出力する(前置きやマークダウンは付けない)"""


# ---------- 切り出し ----------
def make_crop(row):
    """PDF の指定ページを dpi で描き、px 座標で切り出す(座標が空ならページ全体)"""
    out = WORK / "crops" / f"{row.id}.png"
    if out.exists():
        return out
    with pymupdf.open(PDF_DIR / row.election / f"{row.doc}.pdf") as d:
        page = d[int(row.page) - 1]
        clip = None
        if str(row.x0).strip():
            k = 72 / float(row.dpi)                  # px → PDF の座標(1/72 インチ)
            clip = pymupdf.Rect(float(row.x0) * k, float(row.y0) * k, float(row.x1) * k, float(row.y1) * k)
        pix = page.get_pixmap(dpi=int(row.dpi), clip=clip)
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out)
    return out


# ---------- 読み取り ----------
def run_tesseract(img_path):
    cmd = os.environ.get("TESSERACT_CMD", "tesseract")   # Windows で PATH が通っていなければ .env で場所を指定
    r = subprocess.run([cmd, str(img_path), "stdout", "-l", "jpn", "--psm", "3"],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return r.stdout, {}


def run_claude(img_path, cfg, client):
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((cfg["max_edge"], cfg["max_edge"]))     # 長い辺を上限以下に(縦横比は保つ)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    msg = client.messages.create(
        model=cfg["model"], max_tokens=4096, 
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    cost = (u.input_tokens * cfg["price_in"] + u.output_tokens * cfg["price_out"]) / 1e6
    return text, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                  "cost_usd": round(cost, 5), "image_size": f"{im.width}x{im.height}"}


# ---------- 採点 ----------
def normalize(s):
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", s))


def edit_distance(a, b):
    """a を b に直すのに必要な、1文字の挿入・削除・置換の最小回数(レーベンシュタイン距離)"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(gt, hyp):
    g, h = normalize(gt), normalize(hyp)
    errors = edit_distance(h, g)                       # 直すのに必要な回数(間違いの数)
    # 順番を無視した文字の一致(枠の読む順番が違うだけの差を、CER と分けて見るため)
    common = sum((Counter(g) & Counter(h)).values())
    return {
        "gt_chars": len(g), "ocr_chars": len(h), "errors": errors,
        "cer": round(errors / max(len(g), 1), 4),
        "char_recall": round(common / max(len(g), 1), 4),      # 正解の文字のうち、拾えた割合
        "char_precision": round(common / max(len(h), 1), 4),   # 読んだ文字のうち、正解にある割合
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--force", action="store_true", help="読み取り結果があってもやり直す")
    args = ap.parse_args()

    # .env を最初に読む(ANTHROPIC_API_KEY と TESSERACT_CMD を環境変数として使えるようにする)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    client = None
    if any(METHODS[m]["kind"] == "claude" for m in args.methods):
        import anthropic
        client = anthropic.Anthropic()                    # ANTHROPIC_API_KEY を環境変数から読む

    pages = pd.read_csv(PAGES, dtype=str).fillna("")
    rows, total_cost = [], 0.0
    for row in pages.itertuples():
        img = make_crop(row)
        gt_path = GT_DIR / f"{row.id}.txt"
        gt = gt_path.read_text(encoding="utf-8") if gt_path.exists() else ""
        if not normalize(gt):
            print(f"[正解なし] {row.id}  ← {gt_path.name} に正解を打ち込むと採点される")

        for m in args.methods:
            cfg = METHODS[m]
            out = WORK / "out" / m / f"{row.id}.txt"
            meta = {}
            if out.exists() and not args.force:
                hyp = out.read_text(encoding="utf-8")
            else:
                t0 = time.time()
                hyp, meta = run_tesseract(img) if cfg["kind"] == "tesseract" else run_claude(img, cfg, client)
                meta["seconds"] = round(time.time() - t0, 1)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(hyp, encoding="utf-8")
                total_cost += meta.get("cost_usd", 0)
            rec = {"id": row.id, "method": m, **meta}
            if normalize(gt):
                rec.update(score(gt, hyp))
                print(f"{row.id:<12} {m:<14} CER {rec['cer']:.3f}  拾えた割合 {rec['char_recall']:.3f}")
            rows.append(rec)

    df = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT, index=False, encoding="utf-8")
    if "cer" in df:
        s = df.dropna(subset=["cer"])
        # 方法ごとの全体の CER = 全ページの間違いの合計 ÷ 正解の文字数の合計(短いページに引っぱられないように)
        # errors は score() で丸める前の回数を記録してあるので、そのまま足せる
        summary = s.groupby("method")[["errors", "gt_chars"]].sum()
        summary["cer_total"] = (summary["errors"] / summary["gt_chars"]).round(4)
        print("\n方法ごとの CER(全ページ合計)")
        print(summary["cer_total"].sort_values().to_string())
    if total_cost:
        print(f"\n今回の API 料金の目安: ${total_cost:.4f}")
    print(f"保存: {REPORT}")


if __name__ == "__main__":
    main()

"""OCR の本番: 画像だけの重点公約のページを Claude Sonnet 5 で読み、ページの本文を埋める。

12_ocr_eval.py の比較で一番間違いが少なかった Claude Sonnet 5 を使う。
比較では画像を「300dpi のまま縮小せずに」渡して CER 0.001 だった。本番も同じ条件にするため、
長い辺が 2576px(Sonnet が縮小せずに受け取れる上限)を超えるページは、文字のない白い帯で
タイルに切ってから渡す。維新2026 のスライドだけは、比較でページ全体を縮小して渡しても
間違いゼロだったので、切らずにそのまま渡す。

使い方(プロジェクトのフォルダで):
    python scripts/12_ocr.py --preview                          # 切り方を画像で確認するだけ(API は使わない・無料)
    python scripts/12_ocr.py --preview --doc 2025_upper/reiwa --pages 13 40
    python scripts/12_ocr.py --doc 2025_upper/reiwa --pages 13 40   # 一部のページだけ読む
    python scripts/12_ocr.py                                    # 全部読む(読んだタイルは再利用されるので、途中で止めても続きから)

入力:
    data/interim/manifestos/pages.parquet   11_extract_text.py の出力(needs_ocr のページが対象)
    data/raw/manifestos/<election>/<doc>.pdf
出力:
    data/interim/manifestos/ocr/tiles/<election>/<doc>/p<page>_t<n>.png    タイル画像
    data/interim/manifestos/ocr/text/<election>/<doc>/p<page>_t<n>.txt     タイルごとの読み取り結果(あれば再利用)
    data/interim/manifestos/ocr/preview/<election>_<doc>_p<page>.png       切り方の確認用(赤い枠と読む順番)
    data/interim/manifestos/pages_ocr.parquet   pages.parquet の OCR ページに本文を入れたもの(text_source 列: pdf / ocr)
    reports/ocr_tiles.csv                        1行 = 1タイル(位置・縮小したか・トークン数・料金)
"""
import argparse
import importlib
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ev = importlib.import_module("12_ocr_eval")     # 比較と同じ PROMPT・モデル設定・呼び出し方を使う

PAGES = ROOT / "data" / "interim" / "manifestos" / "pages.parquet"
PDF_DIR = ROOT / "data" / "raw" / "manifestos"
OCR_DIR = ROOT / "data" / "interim" / "manifestos" / "ocr"
OUT_PAGES = ROOT / "data" / "interim" / "manifestos" / "pages_ocr.parquet"
REPORT = ROOT / "reports" / "ocr_tiles.csv"

METHOD = "claude-sonnet"
DPI = 300
LIMIT = ev.METHODS[METHOD]["max_edge"]   # 2576px。これ以下のタイルなら縮小されない

# OCR する文書と渡し方。表紙・白紙・まとめだけのページ(共産・公明・みらい・中道)は対象外
OCR_DOCS = {
    ("2025_upper", "ishin"): "spread",   # A3 見開き。折り目で左右に分けてから切る
    ("2025_upper", "reiwa"): "tile",
    ("2025_upper", "sanseito"): "tile",
    ("2026_lower", "hoshu"): "tile",
    ("2026_lower", "ishin"): "whole",    # スライド。比較でページ全体を縮小して渡し、間違いゼロを確認済み
    ("2026_lower", "reiwa"): "tile",
    ("2026_lower", "sdp"): "tile",
}
# OCR しないページ(今は無し)。参政2025 の1枚目は機関紙の表紙だが、与党批判の形で政策の立場を
# 7項目挙げているので読んでおく。分析に使うかは分類の段階で全党共通の決まりで決める(manifesto_sources.csv の note)
SKIP_PAGES = {}

# タイルの切り方の設定(px は 300dpi のとき)
EDGE = 40        # 隣の画素との明るさの差がこれより大きければ「文字や線がある」とみなす(なだらかな背景色は無視される)
NOISE = 3        # 1行(1列)にこの数以下の点しかなければ空白とみなす(スキャンの細かい汚れ対策)
GAP_ROW = 10     # 横に切るのに必要な空白の帯の高さ(行間)
GAP_COL = 40     # 縦に切るのに必要な空白の帯の幅(段と段の間)
PAD = 20         # タイルの周りに残す余白(切る前の範囲からははみ出さない)
MIN_PART = 200   # 切ったあとの両側がこれより小さくなる切り方はしない(ページ番号だけのタイルなどを作らない)


# ---------- タイルに切る ----------
def ink_mask(img):
    """文字・線・写真のある画素を True にする(明るさが急に変わる所)"""
    g = np.asarray(img.convert("L"), dtype=np.int16)
    m = np.zeros(g.shape, dtype=bool)
    m[:, 1:] |= np.abs(np.diff(g, axis=1)) > EDGE
    m[1:, :] |= np.abs(np.diff(g, axis=0)) > EDGE
    return m


def blank_bands(profile, min_len):
    """profile(各行または各列の点の数)から、空白が min_len 以上続く帯を (始まり, 終わり) で返す"""
    blank = profile <= NOISE
    bands, start = [], None
    for i, b in enumerate(np.append(blank, False)):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_len:
                bands.append((start, i))
            start = None
    return bands


def shrink(m, box):
    """box を中身(点のある範囲)まで縮め、PAD の余白を付ける。中身がなければ None"""
    x0, y0, x1, y1 = box
    sub = m[y0:y1, x0:x1]
    rows, cols = np.where(sub.sum(axis=1) > NOISE)[0], np.where(sub.sum(axis=0) > NOISE)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (max(x0, x0 + cols[0] - PAD), max(y0, y0 + rows[0] - PAD),
            min(x1, x0 + cols[-1] + 1 + PAD), min(y1, y0 + rows[-1] + 1 + PAD))


def pick_cut(bands, start, length):
    """空白の帯の中から切る位置を選ぶ: 最初の部分が LIMIT 以下に収まる範囲で、いちばん遠い位置
    (1枚目にできるだけ多く入れる。等分を目指すと、見出しと番号のような近くにある組を分けやすい)"""
    cands = [(a + b) // 2 for a, b in bands
             if start + MIN_PART <= (a + b) // 2 <= min(start + LIMIT, start + length - MIN_PART)]
    return max(cands) if cands else None


def split(m, box, depth=0):
    """box を LIMIT 以下のタイルに分け、読む順番(上→下、左→右。段組みは段ごと)に並べて返す。
    返り値: [(x0, y0, x1, y1, scaled)]。scaled=True は空白の帯が見つからず縮小して渡すタイル"""
    box = shrink(m, box)
    if box is None:
        return []
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= LIMIT and h <= LIMIT:
        return [(x0, y0, x1, y1, False)]
    if depth > 20:
        return [(x0, y0, x1, y1, True)]

    sub = m[y0:y1, x0:x1]
    col_bands = blank_bands(sub.sum(axis=0), GAP_COL)
    cut_x = None
    if w > LIMIT:                                    # 横に長すぎる → 段の間で縦に切るしかない
        c = pick_cut(col_bands, 0, w)
        cut_x = x0 + c if c is not None else None
    else:                                            # 縦に長いだけ → 段組みなら先に段で分ける(読む順番を守るため)
        gutters = [(a, b) for a, b in col_bands if a > w * 0.25 and b < w * 0.75]   # 端の余白は段の間ではない
        if gutters:
            a, b = max(gutters, key=lambda ab: ab[1] - ab[0])
            cut_x = x0 + (a + b) // 2
    if cut_x is not None:
        return split(m, (x0, y0, cut_x, y1), depth + 1) + split(m, (cut_x, y0, x1, y1), depth + 1)
    if h > LIMIT:                                    # 行間で横に切る
        c = pick_cut(blank_bands(sub.sum(axis=1), GAP_ROW), 0, h)
        if c is not None:
            return split(m, (x0, y0, x1, y0 + c), depth + 1) + split(m, (x0, y0 + c, x1, y1), depth + 1)
    return [(x0, y0, x1, y1, True)]                  # 切れる場所がない → このまま渡す(API 側で縮小される)


def render(election, doc, page):
    with pymupdf.open(PDF_DIR / election / f"{doc}.pdf") as d:
        pix = d[page - 1].get_pixmap(dpi=DPI)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def make_tiles(img, mode):
    W, H = img.size
    if mode == "whole":
        return [(0, 0, W, H, max(W, H) > LIMIT)]
    m = ink_mask(img)
    if mode == "spread":                             # 見開き: まず折り目(真ん中)で左右のページに分ける
        return split(m, (0, 0, W // 2, H)) + split(m, (W // 2, 0, W, H))
    return split(m, (0, 0, W, H))


def save_preview(img, tiles, out):
    k = 0.2
    pv = img.resize((int(img.width * k), int(img.height * k)))
    dr = ImageDraw.Draw(pv)
    for i, (x0, y0, x1, y1, scaled) in enumerate(tiles, 1):
        dr.rectangle([x0 * k, y0 * k, x1 * k, y1 * k], outline=(255, 160, 0) if scaled else (255, 0, 0), width=3)
        dr.text((x0 * k + 6, y0 * k + 4), str(i), fill=(255, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    pv.save(out)


# ---------- 本文をそろえる ----------
PAGE_NUM = re.compile(r"^[\d\-–—/ ]{1,7}$")   # 11_extract_text.py と同じ(NFKC の後なので半角だけでよい)
# 文字のないタイルを渡すと、Claude は書き起こしの代わりに「文字がない」という説明を返すことがある
NO_TEXT = ("申し訳", "文字が含まれて", "判読できる文字", "書き起こせる内容", "書き起こす文字")


def is_no_text(text):
    """Claude が書き起こしの代わりに「文字がない」と説明した出力か(本文の「申し訳」で誤って捨てないよう、短い・画像に触れている、も条件にする)"""
    return len(text) < 400 and "画像" in text and any(k in text for k in NO_TEXT)


def clean(text):
    """11_extract_text.py と同じ掃除: NFKC → 前後の空白を除く → 空行とページ番号だけの行を除く"""
    lines = [unicodedata.normalize("NFKC", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l and not PAGE_NUM.match(l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="切り方の確認画像だけ作る(API は使わない)")
    ap.add_argument("--doc", nargs="+", help="対象を絞る(例: 2025_upper/reiwa)")
    ap.add_argument("--pages", nargs="+", type=int, help="対象のページ番号を絞る")
    ap.add_argument("--force", action="store_true", help="読み取り結果があってもやり直す")
    args = ap.parse_args()

    pages = pd.read_parquet(PAGES)
    target = pages[(pages["kind"] == "main") & pages["needs_ocr"]]
    target = target[[(r.election, r.doc) in OCR_DOCS and r.page not in SKIP_PAGES.get((r.election, r.doc), set())
                     for r in target.itertuples()]]
    if args.doc:
        target = target[(target["election"] + "/" + target["doc"]).isin(args.doc)]
    if args.pages:
        target = target[target["page"].isin(args.pages)]
    print(f"対象 {len(target)} ページ")

    # 前回までのタイルの記録。読み取り結果を使い回すのは「同じ範囲のタイル」のときだけ
    # (切り方を変えると p005_t2 が別の範囲を指すことがあるため、ファイル名だけでは判断しない)
    old = pd.read_csv(REPORT) if REPORT.exists() else pd.DataFrame()
    prev = {(rec["election"], rec["doc"], int(rec["page"]), int(rec["tile"])): rec
            for rec in old.to_dict("records")}

    client = None
    if not args.preview:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(ROOT / ".env")
        client = anthropic.Anthropic()
    cfg = ev.METHODS[METHOD]

    log, total_cost = [], 0.0
    try:
        for r in target.itertuples():
            img = render(r.election, r.doc, r.page)
            tiles = make_tiles(img, OCR_DOCS[(r.election, r.doc)])
            n_scaled = sum(t[4] for t in tiles)
            print(f"{r.election} {r.doc:<9} p{r.page:<3} {img.width}x{img.height} → タイル {len(tiles)}"
                  + (f"(うち縮小 {n_scaled})" if n_scaled else ""))
            if args.preview:
                save_preview(img, tiles, OCR_DIR / "preview" / f"{r.election}_{r.doc}_p{r.page:03d}.png")
                continue

            page_log = []                              # このページの記録(全タイルを読み終えたら log に入れる)
            for i, (x0, y0, x1, y1, scaled) in enumerate(tiles, 1):
                name = f"p{r.page:03d}_t{i}"
                tile_png = OCR_DIR / "tiles" / r.election / r.doc / f"{name}.png"
                out = OCR_DIR / "text" / r.election / r.doc / f"{name}.txt"
                rec = {"election": r.election, "doc": r.doc, "page": r.page, "tile": i,
                       "x0": x0, "y0": y0, "x1": x1, "y1": y1, "scaled": scaled}
                p = prev.get((r.election, r.doc, r.page, i))
                same = p is not None and (p["x0"], p["y0"], p["x1"], p["y1"]) == (x0, y0, x1, y1)
                if out.exists() and same and not args.force:
                    rec = {**p, **rec}                   # 前回の料金・時間の記録を引き継ぐ
                else:
                    if out.exists() and not same:
                        why = "前回の記録がない(途中で止まったページ)" if p is None else "前回と切り方が違う"
                        print(f"  [読み直し] {name}: {why}")
                    tile_png.parent.mkdir(parents=True, exist_ok=True)
                    img.crop((x0, y0, x1, y1)).save(tile_png)
                    t0 = time.time()
                    text, meta = ev.run_claude(tile_png, cfg, client)
                    meta["seconds"] = round(time.time() - t0, 1)
                    if meta["output_tokens"] >= 4096:
                        print(f"  [注意] {name}: 出力が上限 4096 トークンに達した(途中で切れている可能性)")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(text, encoding="utf-8")
                    total_cost += meta["cost_usd"]
                    rec.update(meta)
                page_log.append(rec)
            log.extend(page_log)                           # 途中で止まったページは記録しない(半分だけで「そろった」と判定されないように)
    finally:
        # エラー(残高切れなど)や Ctrl+C で止まっても、読み終えたページの記録は必ず書く
        # (書かないと、次の実行で読み終えたタイルを「記録なし」として読み直し、料金を二重に払う)
        if log and not args.preview:
            # 今回処理したページは、今回のタイルの記録で丸ごと置き換える(古い切り方のタイルの行を残さない)
            # 今回処理しなかったページ(--doc や --pages で外したページ)の記録はそのまま残す
            new = pd.DataFrame(log)
            if len(old):
                done = set(zip(new["election"], new["doc"], new["page"]))
                keep = [(e, d, pg) not in done for e, d, pg in zip(old["election"], old["doc"], old["page"])]
                new = pd.concat([old[keep], new])
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            new.sort_values(["election", "doc", "page", "tile"]).to_csv(REPORT, index=False, encoding="utf-8")

    if args.preview:
        print(f"確認画像: {OCR_DIR / 'preview'}(赤 = そのまま、橙 = 切れず縮小して渡すタイル。数字は読む順番)")
        return

    # pages.parquet の OCR ページに本文を入れる(全タイルが読めたページだけ)
    # 何枚のタイルに切ったかは reports/ocr_tiles.csv の記録で決める。t1〜tN が全部そろったページだけ入れ、
    # 途中で止まったページや、古い切り方の余分なファイル(t{N+1} 以降)は使わない
    tbl = pd.read_csv(REPORT) if REPORT.exists() else pd.DataFrame(columns=["election", "doc", "page"])
    n_tiles = tbl.groupby(["election", "doc", "page"]).size().to_dict()
    out_pages = pages.copy()
    out_pages["text_source"] = "pdf"
    filled = missing = 0
    for idx, r in pages.iterrows():
        if (r["election"], r["doc"]) not in OCR_DOCS or not r["needs_ocr"] or r["kind"] != "main":
            continue
        if r["page"] in SKIP_PAGES.get((r["election"], r["doc"]), set()):
            continue
        n = n_tiles.get((r["election"], r["doc"], r["page"]), 0)
        files = [OCR_DIR / "text" / r["election"] / r["doc"] / f"p{r['page']:03d}_t{i}.txt" for i in range(1, n + 1)]
        if n == 0 or not all(f.exists() for f in files):
            missing += 1
            continue
        texts = []
        for f in files:
            t = f.read_text(encoding="utf-8")
            if is_no_text(t):
                print(f"[文字なしとして除外] {r['election']}/{r['doc']}/{f.name}: {t[:40]!r}…")
                continue
            texts.append(t)
        text = clean("\n".join(texts))
        out_pages.at[idx, "text"] = text
        out_pages.at[idx, "n_chars"] = len(text.replace("\n", ""))
        out_pages.at[idx, "text_source"] = "ocr"
        filled += 1
    out_pages.to_parquet(OUT_PAGES, index=False)
    print(f"\n本文を入れたページ {filled} / まだ読んでいないページ {missing}")
    if total_cost:
        print(f"今回の API 料金の目安: ${total_cost:.3f}")
    print(f"保存: {OUT_PAGES}")


if __name__ == "__main__":
    main()

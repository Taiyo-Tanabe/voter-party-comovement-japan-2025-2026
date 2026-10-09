"""集めた公約(PDF・HTML)から本文を取り出し、ページごとの表と読みやすいテキストを保存する。

使い方(プロジェクトのフォルダで):
    python scripts/11_extract_text.py

入力: data/external/manifesto_sources.csv と、そこに書かれた data/raw/manifestos/ のファイル
出力:
    data/interim/manifestos/pages.parquet   1行 = 1ページ(HTML は1ファイル = 1ページ)
    data/interim/manifestos/text/<election>/<doc>.txt   人が読む用(ページの区切り付き)
    reports/manifesto_inventory.csv         1行 = 1文書の点検表(ページ数・文字数・OCRが要るページ数)

ここでの掃除は「誰がやっても同じ結果になる機械的なもの」だけにする:
    0. 縦書きで1文字ずつ別の行になったものを1行につなげる(表紙の題名など)
    1. 同じ行・前の行を含む行が続いたら1つにまとめる(共産党の PDF は影付きの見出しが2回取れる)
    2. 多くのページに出てくる短い行(ヘッダー・フッター)を除く
    3. ページ番号だけの行を除く
政策項目への分割や、HTML のメニュー部分の除去は 13_segment で行う。
文字が取れないページ(画像だけのページ)は needs_ocr=True として残し、12_ocr で埋める。
"""
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "external" / "manifesto_sources.csv"
OUT_DIR = ROOT / "data" / "interim" / "manifestos"
TEXT_DIR = OUT_DIR / "text"
INVENTORY = ROOT / "reports" / "manifesto_inventory.csv"

MIN_CHARS = 30          # 空白を除いてこれ未満の文字しか取れないページは「画像だけ」とみなす
HEADER_SHARE = 0.5      # 半分以上のページに出てくる行はヘッダー・フッターとみなす
HEADER_MAX_LEN = 60     # ただし長い行は本文の可能性があるので消さない
PAGE_NUM = re.compile(r"^[\d０-９\-–—/／ ]{1,7}$")   # 「6」「- 6 -」「6/24」など
WORD_CHAR = re.compile(r"[\w぀-ヿ一-鿿]")  # 文字(記号ではない)


def count_chars(s):
    return len(re.sub(r"\s", "", s))


def doc_kind(stem):
    """ファイル名から文書の種類を決める: ldp → main、ldp_full → full、ishin_core → core"""
    if stem.endswith("_full"):
        return "full"
    if stem.endswith("_core"):
        return "core"
    return "main"


# ---------- 取り出し ----------
def pdf_pages(path):
    with pymupdf.open(path) as d:
        return [p.get_text() for p in d]


class TextOnly(HTMLParser):
    """HTML から目に見える文字だけを取り出す(script・style などの中身は捨てる)"""
    SKIP = {"script", "style", "noscript", "svg", "template"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_pages(path):
    p = TextOnly()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return ["".join(p.parts)]


# ---------- 掃除 ----------
def clean_lines(text):
    lines = [unicodedata.normalize("NFKC", l).strip() for l in text.splitlines()]
    return [l for l in lines if l]


def join_vertical(lines):
    """縦書きの文字が1文字ずつ別の行になったもの(3行以上続く)を1行につなげる
    例: 「日」「本」「を」「動」「か」「す」 → 「日本を動かす」"""
    out, run = [], []
    for l in lines + [""]:              # 最後に空行を足して、残った run を必ず出す
        if len(l) == 1 and WORD_CHAR.match(l):
            run.append(l)
            continue
        if len(run) >= 3:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if l:
            out.append(l)
    return out


def drop_repeats(lines):
    """同じ行・前の行を含む行が続いたら1つにまとめる(短い行は対象外)
    共産党の PDF は影付きの見出しが2回取れ、2回目に本文がくっついていることがある:
        「を見直すのは当然です」
        「を見直すのは当然です......自民党政治は、消費税」 ← こちらだけ残す
    """
    out = []
    for l in lines:
        if out and len(out[-1]) >= 5 and l.startswith(out[-1]):
            out[-1] = l                 # 同じ行なら置き換えても同じ。長いほうを残す
            continue
        out.append(l)
    return out


def find_headers(pages_lines):
    """半分以上のページに出てくる短い行を探す(4ページ以上の文書だけ)"""
    n = len(pages_lines)
    if n < 4:
        return set()
    seen = Counter(l for lines in pages_lines for l in set(lines) if 4 <= len(l) <= HEADER_MAX_LEN)
    return {l for l, c in seen.items() if c >= max(3, n * HEADER_SHARE)}


def main():
    src = pd.read_csv(SOURCES, dtype=str).fillna("")
    rows, inv = [], []
    for r in src.itertuples():
        path = ROOT / r.file
        if not path.exists():
            print(f"[ファイルなし] {r.file}  ← 10_download_manifestos.py を先に実行")
            continue
        doc = path.stem
        raw = pdf_pages(path) if path.suffix == ".pdf" else html_pages(path)

        pages_lines = [drop_repeats(join_vertical(clean_lines(t))) for t in raw]
        headers = find_headers(pages_lines)
        for i, (t, lines) in enumerate(zip(raw, pages_lines), start=1):
            kept = [l for l in lines if l not in headers and not PAGE_NUM.match(l)]
            text = "\n".join(kept)
            rows.append({
                "election": r.election, "party": r.party, "doc": doc, "kind": doc_kind(doc),
                "source": path.suffix[1:], "page": i,
                "text_raw": t, "text": text,
                "n_chars": count_chars(text),
                "needs_ocr": count_chars(t) < MIN_CHARS,
            })

        d = pd.DataFrame(rows[-len(raw):])
        inv.append({
            "election": r.election, "party": r.party, "doc": doc, "kind": doc_kind(doc),
            "source": path.suffix[1:], "pages": len(raw),
            "ocr_pages": int(d["needs_ocr"].sum()),
            "chars_raw": sum(count_chars(t) for t in raw),
            "chars_clean": int(d["n_chars"].sum()),
            "header_lines_removed": len(headers),
        })
        print(f"{r.election} {r.party:<4} {doc:<12} {len(raw):>4}ページ  "
              f"{inv[-1]['chars_clean']:>7,}字  OCRが要るページ {inv[-1]['ocr_pages']}")

        # 人が読む用のテキスト(ページの区切り付き)
        out = TEXT_DIR / r.election / f"{doc}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(
            f"\n===== {row['page']}ページ{' (要OCR)' if row['needs_ocr'] else ''} =====\n{row['text']}\n"
            for row in d.to_dict("records")), encoding="utf-8")

    pages = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages.to_parquet(OUT_DIR / "pages.parquet", index=False)
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inv).to_csv(INVENTORY, index=False, encoding="utf-8")

    main_docs = pages[pages["kind"] == "main"]
    print(f"\n全 {pages['doc'].count()} ページ(文書 {len(inv)} 件)")
    print(f"重点公約で OCR が要るページ: {int(main_docs['needs_ocr'].sum())}")
    print(f"保存: {OUT_DIR / 'pages.parquet'}")
    print(f"保存: {INVENTORY}")


if __name__ == "__main__":
    main()

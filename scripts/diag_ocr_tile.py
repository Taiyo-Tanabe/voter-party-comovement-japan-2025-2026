"""1枚のタイルを OCR と同じ条件で Claude に読ませ、返事の中身を詳しく調べる(点検用・本番の結果は変えない)。

12_ocr.py は返事のうち「文字(text)」の部品だけを保存するので、それ以外の部品や
返事の終わり方(stop_reason)は記録に残らない。出力が空だったタイルの原因を調べるために使う。

使い方(プロジェクトのフォルダで):
    python scripts/diag_ocr_tile.py data/interim/manifestos/ocr/tiles/2025_upper/sanseito/p001_t2.png

出力:
    画面: stop_reason・部品ごとの種類と長さ・先頭の文字・トークン数
    data/interim/manifestos/ocr/diag/<タイル名>.json   返事をまるごと保存したもの
"""
import base64
import importlib
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ev = importlib.import_module("12_ocr_eval")     # 本番と同じ PROMPT・モデル設定を使う
DIAG_DIR = ROOT / "data" / "interim" / "manifestos" / "ocr" / "diag"


def main():
    if len(sys.argv) != 2:
        sys.exit("使い方: python scripts/diag_ocr_tile.py <タイル画像のパス>")
    tile = Path(sys.argv[1])
    cfg = ev.METHODS["claude-sonnet"]

    from dotenv import load_dotenv
    import anthropic
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()

    # 画像の渡し方は ev.run_claude と同じ
    im = Image.open(tile).convert("RGB")
    im.thumbnail((cfg["max_edge"], cfg["max_edge"]))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    msg = client.messages.create(
        model=cfg["model"], max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": ev.PROMPT},
        ]}],
    )

    print(f"モデル       : {msg.model}")
    print(f"stop_reason  : {msg.stop_reason}")
    print(f"トークン     : 入力 {msg.usage.input_tokens} / 出力 {msg.usage.output_tokens}")
    print(f"部品の数     : {len(msg.content)}")
    for i, b in enumerate(msg.content, 1):
        body = getattr(b, "text", None) or getattr(b, "thinking", None) or ""
        print(f"  部品{i}: 種類 = {b.type} / 長さ {len(body)} 字")
        print(f"         先頭: {body[:200]!r}")
        if len(body) > 200:
            print(f"         末尾: {body[-200:]!r}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAG_DIR / f"{tile.parent.parent.name}_{tile.parent.name}_{tile.stem}.json"
    out.write_text(msg.model_dump_json(indent=2), encoding="utf-8")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

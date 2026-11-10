"""Claude を呼ぶ共通の関数(13_segment の方法比べと、この後の立場の判定で使う)。

OCR(12_ocr.py)で分かったことを最初から入れておく:
    - Sonnet 5 は答える前に「考える(thinking)」ことがあり、その分も出力トークン(料金)に入る。
      max_tokens が小さいと、考えるだけで上限に達して答えが空になる(参政2025 p1 t2)→ 上限を大きくとる
    - 返事の終わり方(stop_reason)と、考えたトークン数を記録する
    - 答え(text の部品)が空、または上限で打ち切られたら、警告を返す(黙って抜けないように)

立場の判定(14_stance.py)で足したこと:
    - system: 毎回同じ前置き(コードブック)を system に入れ、cache=True ならプロンプトキャッシュを使う
      (2回目からは前置きを安く読む。書き込み 1.25倍・読み込み 0.1倍の料金)
"""
import time

# USD / 100万トークン(2026-10 の料金表。12_ocr_eval.py の METHODS と同じ値)
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def make_client():
    """.env の ANTHROPIC_API_KEY を読んで、API の窓口を作る"""
    from pathlib import Path
    from dotenv import load_dotenv
    import anthropic
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return anthropic.Anthropic()


def call_claude(client, model, prompt, max_tokens=16000, effort=None, system=None, cache=False):
    """文章だけの指示を Claude に送り、(答えの文字列, 記録の辞書) を返す。
    記録: stop_reason・入力/出力/考えたトークン数・料金・秒数・警告
    effort: "low" / "medium" / "high" など。None なら API の初期値(Sonnet 5 は high)。
            下げると考える量(= 出力トークン = 料金)が減る
    system: 前置きの文(None なら送らない)。cache=True なら、その前置きをキャッシュする"""
    t0 = time.time()
    extra = {"output_config": {"effort": effort}} if effort else None   # SDK の版によらず送れるよう extra_body で渡す
    kw = {}
    if system is not None:
        block = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        kw["system"] = [block]
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra, **kw,
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    details = getattr(u, "output_tokens_details", None)
    thinking = getattr(details, "thinking_tokens", None) if details is not None else None
    price_in, price_out = PRICES[model]
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    warn = []
    if msg.stop_reason == "max_tokens":
        warn.append(f"上限 {max_tokens} トークンで打ち切り")
    if not text.strip():
        warn.append("答えが空")
    return text, {
        "model": model,
        "stop_reason": msg.stop_reason,
        "input_tokens": u.input_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "output_tokens": u.output_tokens,
        "thinking_tokens": thinking,
        "cost_usd": round((u.input_tokens * price_in + cache_write * price_in * 1.25
                           + cache_read * price_in * 0.1 + u.output_tokens * price_out) / 1e6, 5),
        "seconds": round(time.time() - t0, 1),
        "warning": "・".join(warn),
    }

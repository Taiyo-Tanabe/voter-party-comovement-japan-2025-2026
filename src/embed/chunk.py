"""発言を、BERTに入れられる長さのかたまり(チャンク)に分ける"""
import re

# BERTの上限512から、文の始まりと終わりの印([CLS] [SEP])の2つを引いた数
MAX_TOKENS = 510

# 文の終わり(。?!)の直後で区切る
SENT_END = re.compile(r"(?<=[。?!])")


def split_sentences(text: str) -> list[str]:
    """文章を文に分ける(空の文は除く)"""
    return [s for s in SENT_END.split(text) if s.strip()]


def chunk_ids(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[list[int]]:
    """文の区切りで、max_tokens 以内のかたまりに詰める。
    各かたまりはトークン番号のリスト([CLS] [SEP] はまだ付けない)"""
    chunks, cur = [], []
    for sent in split_sentences(text):
        ids = tokenizer(sent, add_special_tokens=False)["input_ids"]

        # 1文だけで上限を超える場合: 今のかたまりを閉じてから、その文を機械的に切る
        if len(ids) > max_tokens:
            if cur:
                chunks.append(cur)
                cur = []
            for i in range(0, len(ids), max_tokens):
                chunks.append(ids[i:i + max_tokens])
            continue

        # この文を足すと上限を超える場合: 今のかたまりを閉じる
        if len(cur) + len(ids) > max_tokens:
            chunks.append(cur)
            cur = []
        cur = cur + ids

    if cur:
        chunks.append(cur)
    return chunks
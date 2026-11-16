"""C 段階 ⑤: マニフェストの全部の行(policies.parquet の 4,083 行)に、Claude で立場のラベルを付ける。

決まり:
    - プロンプトは 14_stance.py と全く同じもの(試験用で測った v2.1 = コードブック 0.4.1)を使う。
      測った条件と変えると、試験用の κ がこのラベルの質の目安にならなくなるため
    - 14_stance.py と同じく、Claude に渡すのは見出しと文だけ(党・選挙は渡さない)
    - 20行ずつ渡す。組は全部の行を乱数で混ぜて作る(試験用と同じく、色々な党の行が混ざる組にする)
    - ラベルは全部の行に付ける。目次のページ・同じ文の行を外すのは集計のとき(D 段階)
    - 答えが読めなかった組は .bad にして飛ばす。もう一度実行すると、その組だけやり直す

使い方(プロジェクトのフォルダで):
    python scripts/15_stance_all.py          # 料金の見積もりを出すだけ(Claude は呼ばない)
    python scripts/15_stance_all.py --run    # 判定する(途中で止めても、もう一度実行すれば続きから)

出力:
    data/interim/manifestos/stance/all_<tag>/batch_XXX.txt   Claude の答え(保存済みの組は呼ばない)
    data/interim/manifestos/stance/all_<tag>/calls.csv       料金の記録
    data/processed/policies_stance.parquet                   policies.parquet に llm_kind・llm_topic・llm_stance を足した表
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_common import PRICES, call_claude, make_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "interim" / "manifestos" / "policies.parquet"
OUT = ROOT / "data" / "processed" / "policies_stance.parquet"

# 14_stance.py の決まり(プロンプト・答えの読み方・モデル)をそのまま使う。名前が数字で始まるので import 文では読めない
_spec = importlib.util.spec_from_file_location("stance14", Path(__file__).resolve().parent / "14_stance.py")
s14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s14)

TAG = "v2.1"     # 試験用で測った版と同じ
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Claude を呼ぶ(付けないと見積もりだけ)")
    args = ap.parse_args()

    p = pd.read_parquet(POLICIES).reset_index(drop=True)
    p.insert(0, "id", [f"P{i + 1:04d}" for i in range(len(p))])        # 行の番号(policies.parquet の並び順)
    order = p.sample(frac=1, random_state=SEED)                           # 組を作るために混ぜる
    batches = [order.iloc[i:i + s14.BATCH] for i in range(0, len(order), s14.BATCH)]
    system = s14.SYSTEM_HEAD + s14.CODEBOOK.read_text(encoding="utf-8")
    out_dir = s14.STANCE_DIR / f"all_{TAG}"

    # 料金の見積もり(14_stance.py と同じ大まかな数え方)
    p_in, p_out = PRICES[s14.MODEL]
    sys_tok = len(system)
    row_tok = sum(len(s14.build_prompt(b)) for b in batches)
    est = (sys_tok * p_in * 1.25 + sys_tok * p_in * 0.1 * (len(batches) - 1)
           + row_tok * p_in + len(batches) * (3000 + 60 * s14.BATCH) * p_out) / 1e6
    print(f"全 {len(p)} 行・{len(batches)} 回・見積もり 約 ${est:.2f}(tag {TAG})")
    if not args.run:
        print("Claude を呼ぶときは --run を付ける")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client()
    calls_path = out_dir / "calls.csv"
    answers, bad_batches = {}, []
    for k, b in enumerate(batches):
        path = out_dir / f"batch_{k:03d}.txt"
        if not (path.exists() and path.read_text(encoding="utf-8").strip()):
            text, rec = call_claude(client, s14.MODEL, s14.build_prompt(b), effort=s14.EFFORT,
                                    system=system, cache=True)
            path.write_text(text, encoding="utf-8")
            rec.update({"batch": k, "tag": TAG})
            pd.DataFrame([rec]).to_csv(calls_path, mode="a", index=False, header=not calls_path.exists(),
                                       encoding="utf-8-sig")
            print(f"  {k + 1}/{len(batches)} ${rec['cost_usd']} {rec['seconds']}秒 {rec['warning']}")
        try:
            answers.update(s14.parse_answer(path.read_text(encoding="utf-8"), list(b.id)))
        except ValueError as e:                                           # 答えが読めない組は飛ばして、あとでやり直す
            path.replace(path.with_suffix(".bad"))
            bad_batches.append(k)
            print(f"  組 {k} の答えが読めない({e})→ .bad にした。もう一度実行するとやり直す")

    if bad_batches:
        print(f"\n読めなかった組 {len(bad_batches)} 個: {bad_batches}。もう一度 --run を実行する(表はまだ保存しない)")
        return

    df = p.join(pd.DataFrame.from_dict(answers, orient="index"), on="id")
    assert df.llm_kind.notna().all(), "答えの無い行がある"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n保存: {OUT}({len(df)} 行)")
    print(f"決まりに合わない答え: {int((df.llm_bad != '').sum())} 行")
    print(df.llm_kind.value_counts().to_string())
    c = pd.read_csv(calls_path, encoding="utf-8-sig")
    print(f"料金の合計 ${c[c.tag == TAG].cost_usd.sum():.3f}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import sacrebleu


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.preds))

    predictions = [r["prediction"] for r in rows]
    references = [[r["target"] for r in rows]]

    bleu = sacrebleu.corpus_bleu(predictions, references)
    chrf = sacrebleu.corpus_chrf(predictions, references)

    print(f"File: {args.preds}")
    print(f"Examples: {len(rows)}")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")


if __name__ == "__main__":
    main()

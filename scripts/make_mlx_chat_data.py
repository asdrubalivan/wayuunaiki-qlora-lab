import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """Eres un asistente de traducción.
Traduces del español al wayuunaiki.
Devuelve únicamente la traducción, sin explicaciones.
"""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_mlx_chat(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Traduce al wayuunaiki.\n\n"
                            f"Español: {r['source']}\n"
                            "Wayuunaiki:"
                        ),
                    },
                    {"role": "assistant", "content": r["target"]},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/mlx_es_to_guc")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "valid", "test"]:
        src = splits_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}.jsonl"
        rows = read_jsonl(src)
        write_mlx_chat(dst, rows)
        print(f"Wrote {len(rows)} rows to {dst}")


if __name__ == "__main__":
    main()

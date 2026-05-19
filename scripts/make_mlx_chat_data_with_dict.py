import argparse
import csv
import json
from pathlib import Path

from rapidfuzz import fuzz


SYSTEM_PROMPT = """Eres un asistente de traducción.
Traduces del español al wayuunaiki.
Usa las entradas de diccionario cuando sean relevantes.
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


def load_dictionary(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spanish = (row.get("spanish") or "").strip()
            wayuunaiki = (row.get("wayuunaiki") or "").strip()
            if spanish and wayuunaiki:
                entries.append({"spanish": spanish, "wayuunaiki": wayuunaiki})
    return entries


def retrieve_entries(sentence: str, entries: list[dict], k: int = 8) -> list[dict]:
    scored = []
    sent_lower = sentence.lower()
    for entry in entries:
        term = entry["spanish"].lower()
        if term in sent_lower:
            score = 100
        else:
            score = fuzz.partial_ratio(term, sent_lower)
        if score >= 75:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def format_dictionary(entries: list[dict]) -> str:
    if not entries:
        return "No hay entradas relevantes."
    return "\n".join(f"- {e['spanish']} → {e['wayuunaiki']}" for e in entries)


def write_mlx_chat(path: Path, rows: list[dict], dictionary: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            entries = retrieve_entries(r["source"], dictionary, k=8)
            dict_context = format_dictionary(entries)
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Traduce al wayuunaiki usando el diccionario solo si ayuda.\n\n"
                            f"Diccionario relevante:\n{dict_context}\n\n"
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--valid", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dict", default="data/dict/wayuunaiki_dictionary.csv")
    args = parser.parse_args()

    dictionary = load_dictionary(Path(args.dict))
    print(f"Loaded {len(dictionary)} dictionary entries")

    out_dir = Path(args.out_dir)

    train_rows = read_jsonl(Path(args.input))
    write_mlx_chat(out_dir / "train.jsonl", train_rows, dictionary)
    print(f"Wrote {len(train_rows)} train rows")

    if args.valid:
        valid_rows = read_jsonl(Path(args.valid))
        write_mlx_chat(out_dir / "valid.jsonl", valid_rows, dictionary)
        print(f"Wrote {len(valid_rows)} valid rows")


if __name__ == "__main__":
    main()

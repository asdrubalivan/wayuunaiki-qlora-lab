import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from mlx_lm import load, generate
from rapidfuzz import fuzz
from tqdm import tqdm


SYSTEM_PROMPT = """Eres un asistente de traducción.
Traduces del español al wayuunaiki.
Usa las entradas de diccionario cuando sean relevantes.
Devuelve únicamente la traducción, sin explicaciones.
"""


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_prompt(tokenizer, source: str, dict_entries: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Traduce al wayuunaiki usando el diccionario solo si ayuda.\n\n"
                f"Diccionario relevante:\n{format_dictionary(dict_entries)}\n\n"
                f"Español: {source}\n"
                "Wayuunaiki:"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_prediction(text: str) -> str:
    text = text.strip()
    for marker in ["Wayuunaiki:", "Traducción:", "Respuesta:"]:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--dict", default="data/dict/wayuunaiki_dictionary.csv")
    parser.add_argument("--input", default="data/processed/test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    dictionary = load_dictionary(Path(args.dict))
    print(f"Loaded {len(dictionary)} dictionary entries")

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    rows = list(read_jsonl(Path(args.input)))
    if args.limit:
        rows = rows[:args.limit]

    outputs = []
    for row in tqdm(rows):
        entries = retrieve_entries(row["source"], dictionary, k=8)
        prompt = build_prompt(tokenizer, row["source"], entries)
        pred = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        outputs.append({
            **row,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "dictionary_entries": entries,
            "prediction": clean_prediction(pred),
        })

    write_jsonl(Path(args.out), outputs)
    print(f"Wrote {len(outputs)} predictions to {args.out}")


if __name__ == "__main__":
    main()

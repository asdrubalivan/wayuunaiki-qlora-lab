import argparse
import json
from pathlib import Path

from mlx_lm import load, generate
from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(tokenizer, source: str) -> str:
    messages = [
        {"role": "system", "content": "Eres un asistente de traducción. Traduces del español al wayuunaiki. Devuelve únicamente la traducción, sin explicaciones."},
        {"role": "user", "content": f"Traduce al wayuunaiki.\n\nEspañol: {source}\nWayuunaiki:"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_prediction(text: str) -> str:
    text = text.strip()
    for marker in ["Wayuunaiki:", "Traducción:", "Respuesta:", "Assistant:"]:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--input", default="data/processed/test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    rows = read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[:args.limit]

    outputs = []
    for row in tqdm(rows):
        prompt = build_prompt(tokenizer, row["source"])
        pred = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        outputs.append({
            **row,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "prediction": clean_prediction(pred),
        })

    write_jsonl(Path(args.out), outputs)
    print(f"Wrote {len(outputs)} predictions to {args.out}")


if __name__ == "__main__":
    main()

import argparse
import json
import unicodedata
from pathlib import Path


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = " ".join(text.strip().split())
    return text


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--src-lang", default="es")
    parser.add_argument("--tgt-lang", default="guc")
    args = parser.parse_args()

    src_lines = read_lines(Path(args.src))
    tgt_lines = read_lines(Path(args.tgt))

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {len(src_lines)} vs {len(tgt_lines)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines), start=1):
            src = clean_text(src)
            tgt = clean_text(tgt)
            if not src or not tgt:
                continue
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            item = {
                "id": f"{args.src_lang}-{args.tgt_lang}-{i}",
                "source_lang": args.src_lang,
                "target_lang": args.tgt_lang,
                "source": src,
                "target": tgt,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()

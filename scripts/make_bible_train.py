import json
from pathlib import Path

src = Path("external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish")
sources = (src / "train.source").read_text().splitlines()
es_lines = (src / "train.es").read_text().splitlines()
guc_lines = (src / "train.guc").read_text().splitlines()

out = Path("data/processed/train_bible.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

kept = 0
with out.open("w", encoding="utf-8") as f:
    for i, (domain, es, guc) in enumerate(zip(sources, es_lines, guc_lines), start=1):
        if domain.strip() == "bible":
            item = {"id": f"es-guc-{i}", "source_lang": "es", "target_lang": "guc", "source": es.strip(), "target": guc.strip()}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

print(f"Wrote {kept} bible examples to {out}")

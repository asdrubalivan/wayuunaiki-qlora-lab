# Wayuunaiki QLoRA Lab

Experimentos de fine-tuning con QLoRA para traducción automática **español → wayuunaiki** (guc), corriendo en Apple Silicon con [MLX](https://github.com/ml-explore/mlx).

El wayuunaiki es la lengua del pueblo Wayuu, hablada en la península de La Guajira (Colombia y Venezuela). Es una lengua de muy bajos recursos digitales, lo que hace que la traducción automática sea un problema abierto y desafiante.

---

## Modelos y datos en HuggingFace

Los pesos de los adaptadores y los datasets de entrenamiento están alojados en HuggingFace:

| Recurso | URL |
|---|---|
| Adaptadores LoRA | [asdrubalivan/wayuunaiki-qwen25-lora](https://huggingface.co/asdrubalivan/wayuunaiki-qwen25-lora) |
| Datasets | [asdrubalivan/wayuunaiki-dataset](https://huggingface.co/datasets/asdrubalivan/wayuunaiki-dataset) |

```bash
# Descargar adaptadores
hf download asdrubalivan/wayuunaiki-qwen25-lora --local-dir adapters/

# Descargar datasets
hf download asdrubalivan/wayuunaiki-dataset --repo-type dataset --local-dir data/
```

---

## Estructura del repositorio

```
wayuunaiki-qlora-lab/
├── adapters/                        # Configs LoRA (pesos en HuggingFace)
│   ├── qwen25_15b_es_to_guc_run1/   # 300 iters, corpus completo
│   ├── qwen25_15b_es_to_guc_run3/   # 1 000 iters, corpus completo
│   ├── qwen25_dict/                 # 1 000 iters, Tatoeba con dict en prompt
│   ├── qwen25_dict_short/           # 300 iters, Tatoeba corto con dict
│   └── qwen25_dict_short_1k/        # 1 000 iters, Tatoeba corto con dict
├── data/
│   ├── dict/                        # Diccionario español ↔ wayuunaiki (CSV)
│   ├── mlx_es_to_guc/               # Formato chat para MLX-LM, corpus completo
│   ├── mlx_tatoeba_with_dict/       # Formato chat, Tatoeba con dict en prompt
│   ├── mlx_tatoeba_short_with_dict/ # Formato chat, Tatoeba corto con dict
│   ├── processed/                   # Pares paralelos en JSONL
│   └── raw/                         # Datos crudos (ignorados en git)
├── external/                        # Submodules con datasets externos
│   ├── americasnlp2025/             # AmericasNLP 2025 shared task
│   └── norgrai-wayuunaiki/          # Corpus Norgrai + Tatoeba
├── outputs/                         # Predicciones por experimento
├── scripts/                         # Pipelines de datos, generación y evaluación
├── .gitignore
└── requirements.txt
```

---

## Modelo base

[`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit) — cuantizado a 4 bits para Apple Silicon. Solo el 0.085% de los parámetros (1.3M de 1,543M) se entrenan con QLoRA, lo que hace viable el fine-tuning local con ~2.5 GB de RAM.

---

## Resultados

Todos los experimentos se evaluaron sobre 100 ejemplos del dev set de AmericasNLP 2025.

| Experimento | Descripción | BLEU | chrF |
|---|---|---|---|
| Baseline | Zero-shot, sin fine-tuning | 0.06 | 9.38 |
| QLoRA run1 | 300 iters, corpus completo | 0.03 | 11.22 |
| QLoRA run3 | 1 000 iters, corpus completo | 0.01 | 10.43 |
| **Dict RAG** | **Diccionario en prompt, sin LoRA** | **0.12** | **19.12** |
| QLoRA + Dict (prompt inconsistente) | run1 + dict en inferencia, sin dict en training | 0.11 | 15.17 |
| Tatoeba + Dict consistente | 1 000 iters Tatoeba 41k con dict en prompt | 0.01 | 9.43 |
| Tatoeba corto + Dict consistente | 1 000 iters Tatoeba ≤5 palabras con dict | 0.01 | 5.96 |

**Conclusión principal:** el enfoque más simple — RAG léxico con diccionario sin fine-tuning — ganó con chrF 19.12. Más complejidad no siempre es mejor.

**Lecciones aprendidas:**
- El sweet spot del QLoRA fue 300 iteraciones. Más iteraciones empeoraron los resultados.
- Consistencia del prompt importa: un adapter entrenado sin diccionario en el prompt no sabe usarlo en inferencia.
- Tatoeba (41k pares) tiene oraciones largas y complejas que interfieren con el RAG léxico.
- Val loss y chrF pueden divergir: val loss puede bajar mientras la calidad de traducción empeora.
- Limitación fundamental: el corpus es 89% diccionario. Las métricas no generalizan a texto natural.

---

## Tutorial reproducible

Esta sección documenta el proceso completo, incluyendo decisiones de diseño y hallazgos reales.

### 0. Setup

```bash
git clone --recurse-submodules https://github.com/asdrubalivan/wayuunaiki-qlora-lab
cd wayuunaiki-qlora-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Explorar el dataset

El corpus oficial de AmericasNLP 2025 tiene 59,715 pares de entrenamiento y 6,635 de dev:

```bash
wc -l \
  external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/train.es \
  external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/dev.es
```

La distribución por dominio revela una limitación importante:

```bash
sort external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/train.source \
  | uniq -c | sort -rn
# 53244 dictionary  ← 89% del corpus
#  5470 bible
#   916 books
#    29 putunka-serruma
#    25 constitution
```

El corpus es dominantemente diccionario. Las métricas resultantes reflejan qué tan bien el modelo maneja ese dominio, no texto natural.

### 2. Preparar los datos

Crear los JSONL de entrenamiento y validación:

```bash
python scripts/make_parallel_jsonl.py \
  --src external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/train.es \
  --tgt external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/train.guc \
  --out data/processed/train.jsonl

python scripts/make_parallel_jsonl.py \
  --src external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/dev.es \
  --tgt external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/dev.guc \
  --out data/processed/valid.jsonl

cp data/processed/valid.jsonl data/processed/test.jsonl
```

El script aplica normalización Unicode NFKC y elimina duplicados. Resultado: 59,715 train / 6,635 valid.

Crear subsets por dominio para experimentos más limpios:

```bash
# Solo pares bíblicos del dev (599 ejemplos)
python3 - << 'PY'
import json
from pathlib import Path

src = Path("external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish")
sources = (src / "dev.source").read_text().splitlines()
es_lines = (src / "dev.es").read_text().splitlines()
guc_lines = (src / "dev.guc").read_text().splitlines()

out = Path("data/processed/test_bible.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

kept = 0
with out.open("w", encoding="utf-8") as f:
    for i, (domain, es, guc) in enumerate(zip(sources, es_lines, guc_lines), start=1):
        if domain.strip() == "bible":
            item = {"id": f"es-guc-{i}", "source_lang": "es", "target_lang": "guc",
                    "source": es.strip(), "target": guc.strip()}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

print(f"Wrote {kept} bible examples")
PY

# Solo pares bíblicos del train (5,470 ejemplos)
python scripts/make_bible_train.py
```

Convertir a formato chat para MLX-LM:

```bash
python scripts/make_mlx_chat_data.py \
  --splits-dir data/processed \
  --out-dir data/mlx_es_to_guc
```

### 3. Experimento 1 — Baseline

Correr el modelo base sin fine-tuning sobre 100 ejemplos:

```bash
python scripts/baseline_generate.py \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --input data/processed/test.jsonl \
  --out outputs/baseline_qwen25_15b.jsonl \
  --limit 100

python scripts/evaluate_mt.py --preds outputs/baseline_qwen25_15b.jsonl
# BLEU: 0.06 | chrF: 9.38
```

**Patrones de error observados en los primeros 10 ejemplos:**

| Categoría | % aprox. | Descripción |
|---|---|---|
| `COPY_SPANISH` | ~40% | Devuelve el input sin traducir |
| `REPETITION_LOOP` | ~35% | Colapsa en token repetido (`aya aya`, `kuma kuma`) |
| `HALLUCINATED_LANGUAGE` | ~15% | Morfología inventada mezclando raíces españolas (`sangra yi`) |
| `PARTIAL_TRANSLATION` | ~10% | Copia parcial del input |

El repetition loop ocurre porque el modelo no tiene distribución de probabilidad confiable para wayuunaiki: genera un token que "parece" wayuunaiki y ese mismo token se vuelve el más probable siguiente.

### 4. Experimento 2 — QLoRA

Entrenar con 300 iteraciones (sweet spot experimental):

```bash
mlx_lm.lora \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --train \
  --data data/mlx_es_to_guc \
  --adapter-path adapters/qwen25_15b_es_to_guc_run1 \
  --iters 300 \
  --batch-size 1 \
  --num-layers 4 \
  --mask-prompt \
  --grad-checkpoint
# Val loss: 6.615 → 3.771
# Trainable parameters: 0.085% (1.319M/1543.714M)
```

Generar y evaluar:

```bash
python scripts/generate_with_adapter.py \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --adapter-path adapters/qwen25_15b_es_to_guc_run1 \
  --input data/processed/test.jsonl \
  --out outputs/lora_qwen25_15b_run1.jsonl \
  --limit 100

python scripts/evaluate_mt.py --preds outputs/lora_qwen25_15b_run1.jsonl
# BLEU: 0.03 | chrF: 11.22
```

El QLoRA rompió el repetition loop pero la ganancia en chrF fue modesta (+1.84 puntos). Con 1,000 iteraciones (`run3`) los resultados empeoraron (chrF 10.43): rendimientos decrecientes sobre un corpus de diccionario.

### 5. Experimento 3 — Diccionario RAG

Extraer el diccionario directamente del corpus (entradas cortas ≤ 3 palabras):

```bash
python3 - << 'PY'
import json, csv
from pathlib import Path

corpus = Path("external/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish")
sources = (corpus / "train.source").read_text().splitlines()
es_lines = (corpus / "train.es").read_text().splitlines()
guc_lines = (corpus / "train.guc").read_text().splitlines()

pairs = []
for domain, es, guc in zip(sources, es_lines, guc_lines):
    if domain.strip() == "dictionary":
        es, guc = es.strip(), guc.strip()
        if es and guc and len(es.split()) <= 3:
            pairs.append((es, guc))

pairs = list(dict.fromkeys(pairs))
out = Path("data/dict/wayuunaiki_dictionary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("spanish,wayuunaiki\n")
    for es, guc in pairs[:2000]:
        f.write(f'"{es.replace(chr(34), chr(34)*2)}","{guc.replace(chr(34), chr(34)*2)}"\n')
print(f"Wrote {min(len(pairs), 2000)} entries")
PY
```

Generar con retrieval léxico (rapidfuzz):

```bash
python scripts/dict_augmented_generate.py \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --dict data/dict/wayuunaiki_dictionary.csv \
  --input data/processed/test.jsonl \
  --out outputs/dict_augmented_qwen25_15b.jsonl \
  --limit 100

python scripts/evaluate_mt.py --preds outputs/dict_augmented_qwen25_15b.jsonl
# BLEU: 0.12 | chrF: 19.12  ← mejor resultado del laboratorio
```

El diccionario en el prompt casi dobló el chrF del baseline. El retrieval usa coincidencia exacta (score 100) o similitud aproximada con `fuzz.partial_ratio` (umbral ≥ 75), recuperando hasta 8 entradas relevantes por oración.

**Nota sobre contaminación de dominio:** el diccionario se extrajo del mismo corpus de entrenamiento, por lo que parte de la mejora puede reflejar que el modelo reconoce ese estilo, no que generaliza.

### 6. Experimento 4 — QLoRA + Diccionario con prompt consistente

El problema del experimento anterior (QLoRA run1 + dict): el adapter fue entrenado con un prompt sin diccionario, pero en inferencia se le agrega el diccionario. El modelo ve una estructura que nunca vio durante el entrenamiento.

La solución es entrenar con el diccionario ya incluido en el prompt. Se probaron dos variantes con datos de Tatoeba (corpus del repo `norgrai/wayuunaiki`):

**Tatoeba completo (41,499 pares):**

```bash
# Crear datos Tatoeba
python3 - << 'PY'
import json
from pathlib import Path

src = Path("external/norgrai-wayuunaiki/dataset/bitext")
esp = (src / "tatoeba.RAW.esp_train").read_text().splitlines()
guc = (src / "tatoeba.RAW.guc_train").read_text().splitlines()

out = Path("data/processed/train_tatoeba.jsonl")
kept = 0
with out.open("w", encoding="utf-8") as f:
    for i, (es, g) in enumerate(zip(esp, guc), start=1):
        es, g = es.strip(), g.strip()
        if es and g:
            f.write(json.dumps({"id": f"tatoeba-{i}", "source_lang": "es",
                                 "target_lang": "guc", "source": es, "target": g},
                               ensure_ascii=False) + "\n")
            kept += 1
print(f"Wrote {kept} examples")
PY

# Convertir a formato chat con dict en prompt
python scripts/make_mlx_chat_data_with_dict.py \
  --input data/processed/train_tatoeba.jsonl \
  --valid data/processed/test_bible.jsonl \
  --out-dir data/mlx_tatoeba_with_dict

# Entrenar
mlx_lm.lora \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --train \
  --data data/mlx_tatoeba_with_dict \
  --adapter-path adapters/qwen25_dict \
  --iters 1000 --batch-size 1 --num-layers 4 \
  --mask-prompt --grad-checkpoint

# Evaluar
python scripts/dict_augmented_with_adapter.py \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --adapter-path adapters/qwen25_dict \
  --dict data/dict/wayuunaiki_dictionary.csv \
  --input data/processed/test_bible.jsonl \
  --out outputs/tatoeba_dict_lora_bible.jsonl --limit 100

python scripts/evaluate_mt.py --preds outputs/tatoeba_dict_lora_bible.jsonl
# BLEU: 0.01 | chrF: 9.43  ← apenas igual al baseline
```

**Tatoeba corto ≤5 palabras (4,099 pares):**

```bash
# Crear datos filtrados
python3 - << 'PY'
import json
from pathlib import Path

src = Path("external/norgrai-wayuunaiki/dataset/bitext")
esp = (src / "tatoeba.RAW.esp_train").read_text().splitlines()
guc = (src / "tatoeba.RAW.guc_train").read_text().splitlines()

out = Path("data/processed/train_tatoeba_short.jsonl")
kept = 0
with out.open("w", encoding="utf-8") as f:
    for i, (es, g) in enumerate(zip(esp, guc), start=1):
        es, g = es.strip(), g.strip()
        if es and g and len(es.split()) <= 5:
            f.write(json.dumps({"id": f"tatoeba-short-{i}", "source_lang": "es",
                                 "target_lang": "guc", "source": es, "target": g},
                               ensure_ascii=False) + "\n")
            kept += 1
print(f"Wrote {kept} examples")
PY

python scripts/make_mlx_chat_data_with_dict.py \
  --input data/processed/train_tatoeba_short.jsonl \
  --valid data/processed/test_bible.jsonl \
  --out-dir data/mlx_tatoeba_short_with_dict

mlx_lm.lora \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --train \
  --data data/mlx_tatoeba_short_with_dict \
  --adapter-path adapters/qwen25_dict_short_1k \
  --iters 1000 --batch-size 1 --num-layers 4 \
  --mask-prompt --grad-checkpoint

python scripts/dict_augmented_with_adapter.py \
  --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" \
  --adapter-path adapters/qwen25_dict_short_1k \
  --dict data/dict/wayuunaiki_dictionary.csv \
  --input data/processed/test_bible.jsonl \
  --out outputs/tatoeba_short_dict_lora_bible.jsonl --limit 100

python scripts/evaluate_mt.py --preds outputs/tatoeba_short_dict_lora_bible.jsonl
# BLEU: 0.01 | chrF: 5.96  ← peor que el baseline
```

**Por qué Tatoeba dañó el RAG:** Tatoeba tiene oraciones largas y complejas del dominio bíblico/religioso. El adapter aprendió ese estilo e interfirió con el uso limpio del diccionario en inferencia. Las 4,099 oraciones cortas mostraron overfitting claro (val loss subió después de iter 800).

---

## Configuración de LoRA

```json
{
  "num_layers": 4,
  "learning_rate": 1e-5,
  "batch_size": 1,
  "mask_prompt": true,
  "grad_checkpoint": true,
  "max_seq_length": 2048
}
```

Los pesos LoRA (matrices adicionales, sin modificar el modelo base) se guardan en `.safetensors` y están disponibles en HuggingFace.

---

## Datos externos

| Fuente | Pares | Dominio principal |
|---|---|---|
| [AmericasNLP 2025](https://github.com/AmericasNLP/americasnlp2025) | 59,715 train / 6,635 dev | Diccionario (89%), biblia (9%) |
| [Norgrai Wayuunaiki](https://github.com/norgrai/wayuunaiki) | 41,499 | Bíblico/religioso (Tatoeba) |

---

## Requisitos

- Apple Silicon (M1/M2/M3/M4) con macOS 13+
- Python 3.11+
- MLX 0.31+ / MLX-LM 0.31+
- `rapidfuzz`, `sacrebleu`, `tqdm` (ver `requirements.txt`)

Los pesos de los adaptadores (`.safetensors`) y los datos de entrenamiento grandes no están incluidos en este repositorio. Están disponibles en HuggingFace (ver sección **Modelos y datos en HuggingFace**).

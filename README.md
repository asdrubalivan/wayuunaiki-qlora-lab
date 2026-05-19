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

Para descargar los adaptadores:

```bash
# Descargar un adaptador específico
hf download asdrubalivan/wayuunaiki-qwen25-lora --local-dir adapters/
```

Para descargar los datos:

```bash
hf download asdrubalivan/wayuunaiki-dataset --repo-type dataset --local-dir data/
```

---

## Estructura del repositorio

```
wayuunaiki-qlora-lab/
├── adapters/                  # Configuraciones LoRA (pesos en HuggingFace)
│   ├── qwen25_15b_es_to_guc_run1/   # 300 iteraciones
│   ├── qwen25_15b_es_to_guc_run3/   # 1 000 iteraciones
│   ├── qwen25_dict/                 # 1 000 iters, datos con diccionario
│   ├── qwen25_dict_short/           # 300 iters, datos cortos con diccionario
│   └── qwen25_dict_short_1k/        # 1 000 iters, datos cortos con diccionario
├── configs/                   # Configuraciones de entrenamiento (YAML/JSON personalizados)
├── data/
│   ├── dict/                  # Diccionario español ↔ wayuunaiki (CSV)
│   ├── mlx_es_to_guc/         # Datos en formato chat para MLX-LM (train/valid/test)
│   ├── processed/             # Pares paralelos en JSONL (train/valid/test)
│   └── raw/                   # Datos crudos (ignorados en git)
├── external/                  # Submodules con datasets externos
│   ├── americasnlp2025/       # AmericasNLP 2025 shared task
│   └── norgrai-wayuunaiki/    # Corpus Norgrai
├── outputs/                   # Predicciones generadas por cada experimento
├── reports/                   # Análisis y reportes
├── scripts/                   # Pipelines de preprocesamiento, generación y evaluación
├── .gitignore
└── requirements.txt
```

---

## Modelo base

[`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit) — cuantizado a 4 bits para correr eficientemente en GPU unificada (Apple Silicon).

---

## Experimentos

Se probaron cuatro enfoques sobre el mismo conjunto de 100 ejemplos de prueba:

| Experimento | Descripción | BLEU | chrF |
|---|---|---|---|
| Baseline | Zero-shot, sin fine-tuning | 0.06 | 9.38 |
| LoRA run1 | Fine-tuning 300 iters, rank=8 | 0.03 | 11.22 |
| LoRA run3 | Fine-tuning 1 000 iters, rank=8 | 0.01 | 10.43 |
| Dict-augmented | RAG con diccionario, sin LoRA | **0.12** | **19.12** |
| Dict + LoRA run1 | RAG con diccionario + adaptador run1 | 0.11 | 15.17 |

**Observaciones:**
- El enfoque de augmentación con diccionario (RAG léxico) supera claramente a todos los demás, incluso al fine-tuning, duplicando el chrF respecto al baseline.
- El fine-tuning LoRA con pocas iteraciones no logra mejorar el baseline en BLEU, aunque sí hay una leve ganancia en chrF. El modelo es demasiado pequeño y los datos demasiado escasos para que el ajuste se consolide en pocas iteraciones.
- Los valores de BLEU son bajos en términos absolutos, como es de esperarse en un par de lenguas de muy bajos recursos con morfología compleja.

---

## Configuración de LoRA

```json
{
  "fine_tune_type": "lora",
  "lora_parameters": { "rank": 8, "dropout": 0.0, "scale": 20.0 },
  "num_layers": 4,
  "learning_rate": 1e-5,
  "optimizer": "adam",
  "max_seq_length": 2048,
  "mask_prompt": true
}
```

---

## Reproducción

### 1. Instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar los datos

```bash
# Construir pares paralelos desde archivos de texto plano
python scripts/make_parallel_jsonl.py \
  --src <archivo_fuente.es> \
  --tgt <archivo_destino.guc> \
  --out data/processed/train.jsonl

# Convertir a formato chat para MLX-LM
python scripts/make_mlx_chat_data.py \
  --splits-dir data/processed \
  --out-dir data/mlx_es_to_guc
```

### 3. Fine-tuning con MLX-LM

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --data data/mlx_es_to_guc \
  --adapter-path adapters/mi_run \
  --iters 1000 \
  --rank 8 \
  --num-layers 4 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --grad-checkpoint
```

### 4. Generar predicciones

```bash
# Baseline (zero-shot)
python scripts/baseline_generate.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --input data/processed/test.jsonl \
  --out outputs/baseline.jsonl

# Con adaptador LoRA
python scripts/generate_with_adapter.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters/mi_run \
  --input data/processed/test.jsonl \
  --out outputs/lora.jsonl

# Con augmentación de diccionario
python scripts/dict_augmented_generate.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --dict data/dict/wayuunaiki_dictionary.csv \
  --input data/processed/test.jsonl \
  --out outputs/dict_augmented.jsonl
```

### 5. Evaluar (BLEU + chrF)

```bash
python scripts/evaluate_mt.py --preds outputs/mi_experimento.jsonl
```

---

## Datos externos

Los datos de entrenamiento y evaluación provienen de los siguientes submodules (clonar con `--recurse-submodules`):

```bash
git clone --recurse-submodules https://github.com/asdrubalivan/wayuunaiki-qlora-lab
```

| Fuente | Descripción |
|---|---|
| [AmericasNLP 2025](https://github.com/AmericasNLP/americasnlp2025) | Shared task de traducción de lenguas indígenas americanas |
| [Norgrai Wayuunaiki](https://github.com/norgrai/wayuunaiki) | Corpus paralelo español–wayuunaiki |

---

## Requisitos

- Apple Silicon (M1/M2/M3/M4) con macOS 13+
- Python 3.11+
- MLX 0.31+ / MLX-LM 0.31+

Los pesos de los adaptadores (`.safetensors`) y los datos de entrenamiento grandes no están incluidos en este repositorio. Están disponibles en HuggingFace (ver sección **Modelos y datos en HuggingFace**) o se pueden regenerar con los scripts de la sección anterior.

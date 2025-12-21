# HPO_Mapper_P3

**Embedding retrieval + LLM reranking (Ollama)** for mapping `(finding + anatomical_region)` to HPO terms and associated genes.

## Large files (download from Hugging Face)

This repository intentionally does **not** include large artifacts, such as:

- SQLite database containing synonym-enhanced HPO embeddings + gene mappings  
  (e.g. `hpo_genes_with_synonyms.db`)
- HPO JSON export with definitions  
  (e.g. `hp.json` / `hp.owl.json`)

Download these from **Hugging Face** (see the HF page for this project) and provide local paths with:

- `--db_path`
- `--hpo_json_path`

> Tip: keep these files out of git using `.gitignore`.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull models (examples):

```bash
ollama pull nomic-embed-text
ollama pull llama3.3:70b-instruct-fp16
```

## Input format

Each patient JSON should look like:

```json
{
  "subject_id": "P001",
  "data": [
    {"finding": "abdominal pain", "anatomical_region": "abdomen"},
    {"finding": "diarrhoea", "anatomical_region": "gastrointestinal tract"}
  ]
}
```

## Run

```bash
python hpo_mapper_p3.py \
  --patient_json_dir data/patients_json \
  --db_path data/hpo_genes_with_synonyms.db \
  --hpo_json_path data/hp.json \
  --output_dir out/mapped \
  --top_k 10 \
  --min_sim 0.70 \
  --max_workers 8 \
  --embed_model nomic-embed-text \
  --chat_model llama3.3:70b-instruct-fp16
```

Outputs: one CSV per patient in `out/mapped/`.

## Output columns

- `finding`
- `region`
- `hpo_id`
- `hpo_term`
- `matched_term`
- `genes`
- `similarity` (embedding cosine similarity for the chosen candidate)

## Notes

- If the LLM response cannot be parsed into candidate numbers, the script falls back to the **top-1** candidate by similarity.
- The LLM may select multiple candidates; each is written as its own row.

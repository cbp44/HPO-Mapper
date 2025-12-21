# HPO_Mapper_P2

Maps patient findings + anatomical regions to HPO terms (with genes) using a synonym-embedding SQLite database,
then optionally runs an Ollama LLM quality-control step to flag incorrect mappings.

## Large files (download from Hugging Face)

This repository intentionally does **not** include large artifacts such as:
- the SQLite embedding database (e.g. `hpo_genes_with_synonyms.db`)
- large ontology dumps / precomputed embeddings

Download these from **Hugging Face** (see the HF page for this project) and supply the local path via `--db_path`.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running

Install Python deps:

```bash
pip install -r requirements.txt
```

Pull models (examples):

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct
```

## Input format (patient JSON)

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

## Run (mapping only)

```bash
python pipeline_hpo_map_and_qc.py map \
  --patient_json_dir data/patients_json \
  --db_path data/hpo_genes_with_synonyms.db \
  --output_dir out/mapped \
  --threshold 0.76
```

## Run (mapping + QC)

```bash
python pipeline_hpo_map_and_qc.py run \
  --patient_json_dir data/patients_json \
  --db_path data/hpo_genes_with_synonyms.db \
  --output_dir out/mapped \
  --threshold 0.76 \
  --run_qc \
  --qc_model llama3.1:8b-instruct
```

QC outputs are written to: `out/mapped_llm_qc/` and include a `flag` column (`1` = incorrect mapping).

## Notes

- The SQLite DB must include tables:
  - `hpo_synonym_embeddings(hpo_id, hpo_name, term, embedding)`
  - `hpo_gene(hpo_id, genes)`
- The `embedding` field is expected to be a JSON list of floats.

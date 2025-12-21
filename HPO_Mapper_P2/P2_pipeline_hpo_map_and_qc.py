#!/usr/bin/env python3

"""
HPO Mapper + LLM QC Pipeline (Ollama)

WHAT THIS DOES
--------------
1) Reads patient JSON files (each containing a list of findings + anatomical regions)
2) Maps each finding+region to the best HPO term using cosine similarity on precomputed embeddings
3) Writes one CSV per patient with: finding, region, hpo_id, hpo_term, matched_term, genes
4) Optionally runs an LLM-based QC step (via Ollama) to flag incorrect mappings

IMPORTANT: LARGE FILES (DOWNLOAD FROM HUGGING FACE)
--------------------------------------------------
This repo should NOT contain large artifacts like:
- the SQLite database with embeddings (e.g., hpo_genes_with_synonyms.db)
- any large HPO JSON / ontology dumps
- any large precomputed embeddings

Add a note in your GitHub README and keep these files out of git (e.g., via .gitignore).
Users should download them from Hugging Face (or your release bucket) and point --db_path to them.

Ollama Requirements
-------------------
- Install Ollama: https://ollama.com/
- Ensure the Ollama service is running
- Pull the embedding model (default): nomic-embed-text
- For QC, pull/define a QC model (default is set by --qc_model)

Example:
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b-instruct   # or your custom fine-tuned QC model

USAGE
-----
Mapping only:
  python pipeline_hpo_map_and_qc.py map \
    --patient_json_dir data/patients_json \
    --db_path data/hpo_genes_with_synonyms.db \
    --output_dir out/mapped \
    --threshold 0.76

Mapping + QC:
  python pipeline_hpo_map_and_qc.py run \
    --patient_json_dir data/patients_json \
    --db_path data/hpo_genes_with_synonyms.db \
    --output_dir out/mapped \
    --threshold 0.76 \
    --run_qc \
    --qc_model llama3.1:8b-instruct

Notes
-----
- The DB must include tables:
    hpo_synonym_embeddings(hpo_id, hpo_name, term, embedding)
    hpo_gene(hpo_id, genes)
- embedding is expected as a JSON list (string) of floats.
"""

import argparse
import csv
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.linalg import norm

# Third-party deps:
#   pip install ollama pandas numpy
import ollama
import pandas as pd


# -----------------------------
# Logging
# -----------------------------
def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Also log to stdout for convenience
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


# -----------------------------
# DB loading
# -----------------------------
def load_hpo_embeddings(db_path: str):
    """
    Loads synonym-enhanced HPO embeddings into memory.

    NOTE: This DB can be large and should be downloaded externally (e.g., from Hugging Face).
    Keep it out of git and provide instructions in README.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT hpo_id, hpo_name, term, embedding FROM hpo_synonym_embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for hpo_id, hpo_name, term, embedding_str in rows:
        embeddings.append((hpo_id, hpo_name, term, np.array(json.loads(embedding_str), dtype=float)))
    return embeddings


def load_hpo_gene_map(db_path: str):
    """Loads HPO -> genes map into memory."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT hpo_id, genes FROM hpo_gene")
    rows = cursor.fetchall()
    conn.close()

    # genes stored as "GENE1, GENE2, ..."
    return {hpo_id: genes.split(", ") if genes else [] for hpo_id, genes in rows}


# -----------------------------
# Mapping core
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (norm(a) * norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def find_best_hpo_match(
    finding: str,
    region: str,
    hpo_embeddings,
    embed_model: str,
    threshold: float,
):
    """
    Finds the best HPO match using semantic similarity with synonym-enhanced embeddings.
    """
    query_text = f"{finding} in {region}"
    query_embedding = np.array(
        ollama.embeddings(model=embed_model, prompt=query_text)["embedding"],
        dtype=float,
    )

    best_match = None
    best_score = -1.0

    for hpo_id, hpo_name, matched_term, hpo_embedding in hpo_embeddings:
        score = cosine_sim(query_embedding, hpo_embedding)
        if score > best_score:
            best_score = score
            best_match = {
                "hpo_id": hpo_id,
                "hpo_term": hpo_name,         # canonical HPO term
                "matched_term": matched_term, # synonym/term that triggered match
                "score": best_score,
            }

    if best_match and best_score >= threshold:
        return best_match
    return None


def map_one_finding(
    finding: str,
    region: str,
    hpo_embeddings,
    hpo_gene_map,
    embed_model: str,
    threshold: float,
):
    match = find_best_hpo_match(
        finding=finding,
        region=region,
        hpo_embeddings=hpo_embeddings,
        embed_model=embed_model,
        threshold=threshold,
    )
    if match:
        genes = hpo_gene_map.get(match["hpo_id"], [])
        return {
            "finding": finding,
            "region": region,
            "hpo_id": match["hpo_id"],
            "hpo_term": match["hpo_term"],
            "matched_term": match["matched_term"],
            "genes": genes,
            "score": match["score"],
        }

    return {
        "finding": finding,
        "region": region,
        "hpo_id": "NA",
        "hpo_term": "NA",
        "matched_term": "NA",
        "genes": [],
        "score": "",
    }


def process_patient_json(
    patient_file: str,
    output_dir: str,
    hpo_embeddings,
    hpo_gene_map,
    embed_model: str,
    threshold: float,
) -> str:
    """
    Maps a single patient JSON -> patient CSV.
    Returns output CSV path.
    """
    with open(patient_file, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    subject_id = patient_data.get("subject_id") or os.path.splitext(os.path.basename(patient_file))[0]
    logging.info(f"Processing patient: {subject_id} ({os.path.basename(patient_file)})")

    mapped = []
    for entry in patient_data.get("data", []):
        finding = entry.get("finding", "Unknown")
        region = entry.get("anatomical_region", "Unknown")
        mapped.append(
            map_one_finding(
                finding=finding,
                region=region,
                hpo_embeddings=hpo_embeddings,
                hpo_gene_map=hpo_gene_map,
                embed_model=embed_model,
                threshold=threshold,
            )
        )

    out_csv = os.path.join(output_dir, f"{subject_id}_hpo_mapped.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["finding", "region", "hpo_id", "hpo_term", "matched_term", "genes", "score"])
        for row in mapped:
            writer.writerow([
                row["finding"],
                row["region"],
                row["hpo_id"],
                row["hpo_term"],
                row["matched_term"],
                ", ".join(row["genes"]),
                row["score"],
            ])

    logging.info(f"Saved mapping: {out_csv}")
    return out_csv


def run_mapping(
    patient_json_dir: str,
    db_path: str,
    output_dir: str,
    embed_model: str,
    threshold: float,
    max_workers: int,
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Loading embeddings + gene map from SQLite...")
    hpo_embeddings = load_hpo_embeddings(db_path)
    hpo_gene_map = load_hpo_gene_map(db_path)
    logging.info(f"Loaded {len(hpo_embeddings):,} HPO synonym embeddings and {len(hpo_gene_map):,} HPO->gene mappings.")

    json_files = [
        os.path.join(patient_json_dir, f)
        for f in os.listdir(patient_json_dir)
        if f.lower().endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {patient_json_dir}")

    logging.info(f"Found {len(json_files)} patient JSON files. Mapping with {max_workers} workers...")

    out_csvs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_patient_json,
                patient_file,
                output_dir,
                hpo_embeddings,
                hpo_gene_map,
                embed_model,
                threshold,
            )
            for patient_file in json_files
        ]
        for fut in as_completed(futures):
            out_csvs.append(fut.result())

    logging.info("Mapping step complete.")
    return out_csvs


# -----------------------------
# LLM QC step (Ollama)
# -----------------------------
def build_qc_prompt(row) -> str:
    return (
        "Check whether the following HPO mapping is incorrect.\n\n"
        f"Finding: {row.get('finding', '')}\n"
        f"Region: {row.get('region', '')}\n"
        f"HPO ID: {row.get('hpo_id', '')}\n"
        f"HPO Term: {row.get('matched_term', '')}\n\n"
        "Respond with exactly:\n"
        "- '1' if the mapping is incorrect\n"
        "- blank (empty) if the mapping is correct\n"
        "No other text."
    )


def call_ollama_generate(model_name: str, prompt: str) -> str:
    resp = ollama.generate(model=model_name, prompt=prompt)
    return (resp.get("response") or "").strip()


def qc_one_csv(filepath: str, qc_model: str, output_dir: str) -> str:
    df = pd.read_csv(filepath)
    flags = []

    for _, row in df.iterrows():
        prompt = build_qc_prompt(row)
        try:
            model_response = call_ollama_generate(qc_model, prompt)
            flags.append("1" if model_response == "1" else "")
        except Exception as e:
            logging.error(f"QC error in {os.path.basename(filepath)}: {e}")
            flags.append("")

    df["flag"] = flags

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(filepath))
    df.to_csv(out_path, index=False)
    logging.info(f"QC saved: {out_path}")
    return out_path


def run_qc(mapped_csv_dir: str, qc_model: str) -> str:
    """
    Reads all CSVs in mapped_csv_dir and writes QC outputs to <mapped_csv_dir>_llm_qc
    """
    qc_out_dir = mapped_csv_dir.rstrip("/\\") + "_llm_qc"
    os.makedirs(qc_out_dir, exist_ok=True)

    csvs = [
        os.path.join(mapped_csv_dir, f)
        for f in os.listdir(mapped_csv_dir)
        if f.lower().endswith(".csv")
    ]
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in: {mapped_csv_dir}")

    logging.info(f"Running LLM QC with model '{qc_model}' on {len(csvs)} CSV files...")
    for fp in csvs:
        qc_one_csv(fp, qc_model=qc_model, output_dir=qc_out_dir)

    logging.info("QC step complete.")
    return qc_out_dir


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Map findings to HPO + optional LLM QC (Ollama).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # map
    p_map = sub.add_parser("map", help="Run mapping only (JSON -> mapped CSVs).")
    p_map.add_argument("--patient_json_dir", required=True, help="Folder containing patient JSON files.")
    p_map.add_argument("--db_path", required=True, help="Path to SQLite DB with embeddings + gene mappings (download from HF).")
    p_map.add_argument("--output_dir", required=True, help="Output folder for mapped CSVs.")
    p_map.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model name.")
    p_map.add_argument("--threshold", type=float, default=0.76, help="Cosine similarity threshold for accepting a match.")
    p_map.add_argument("--max_workers", type=int, default=8, help="Threads for multi-patient processing.")

    # qc
    p_qc = sub.add_parser("qc", help="Run QC only on an existing mapped CSV folder.")
    p_qc.add_argument("--mapped_csv_dir", required=True, help="Folder containing mapped CSVs.")
    p_qc.add_argument("--qc_model", required=True, help="Ollama model for QC (user must pull/provide).")

    # run (map then qc)
    p_run = sub.add_parser("run", help="Run mapping then QC.")
    p_run.add_argument("--patient_json_dir", required=True, help="Folder containing patient JSON files.")
    p_run.add_argument("--db_path", required=True, help="Path to SQLite DB with embeddings + gene mappings (download from HF).")
    p_run.add_argument("--output_dir", required=True, help="Output folder for mapped CSVs.")
    p_run.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model name.")
    p_run.add_argument("--threshold", type=float, default=0.76, help="Cosine similarity threshold for accepting a match.")
    p_run.add_argument("--max_workers", type=int, default=8, help="Threads for multi-patient processing.")
    p_run.add_argument("--run_qc", action="store_true", help="If set, run QC after mapping.")
    p_run.add_argument("--qc_model", default="llama3.1:8b-instruct", help="Ollama model for QC.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "map":
        run_mapping(
            patient_json_dir=args.patient_json_dir,
            db_path=args.db_path,
            output_dir=args.output_dir,
            embed_model=args.embed_model,
            threshold=args.threshold,
            max_workers=args.max_workers,
        )

    elif args.cmd == "qc":
        # Logging goes into the mapped folder QC output
        setup_logging(args.mapped_csv_dir)
        run_qc(mapped_csv_dir=args.mapped_csv_dir, qc_model=args.qc_model)

    elif args.cmd == "run":
        run_mapping(
            patient_json_dir=args.patient_json_dir,
            db_path=args.db_path,
            output_dir=args.output_dir,
            embed_model=args.embed_model,
            threshold=args.threshold,
            max_workers=args.max_workers,
        )
        if args.run_qc:
            run_qc(mapped_csv_dir=args.output_dir, qc_model=args.qc_model)


if __name__ == "__main__":
    main()

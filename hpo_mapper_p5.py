#!/usr/bin/env python3

"""
HPO_Mapper_P4 — Embedding Retrieval + LLM Reranking (Ollama)

This script maps patient findings + anatomical regions to Human Phenotype Ontology (HPO) terms.

Pipeline
--------
1) Embed the query "finding in region" using an Ollama embedding model (default: nomic-embed-text)
2) Retrieve top-K candidate HPO terms from a synonym-enhanced embedding SQLite DB using cosine similarity
3) Attach term definitions from an HPO JSON file (download separately)
4) Ask an Ollama chat model to select the best candidate(s) among the top-K
5) Add associated genes from the same SQLite DB
6) Deduplicate HPO terms, keeping only the highest similarity score for each unique term
7) Write one CSV per patient JSON

IMPORTANT: LARGE FILES (DOWNLOAD FROM HUGGING FACE)
--------------------------------------------------
Do NOT commit large artifacts to GitHub, such as:
- the SQLite database containing embeddings and gene mappings (e.g., hpo_genes_with_synonyms.db)
- the HPO JSON file (e.g., hp.json / hp.owl.json)

Instead, download them from Hugging Face (or your preferred hosting) and provide paths via CLI flags:
  --db_path /path/to/hpo_genes_with_synonyms.db
  --hpo_json_path /path/to/hp.json

Ollama Requirements
-------------------
- Install Ollama: https://ollama.com/
- Ensure Ollama is running
- Pull the embedding model (default): nomic-embed-text
- Pull the chat model you want to use for candidate selection (default provided by --chat_model)

Example:
  ollama pull nomic-embed-text
  ollama pull llama3.3:70b-instruct-fp16

Usage
-----
python hpo_mapper_p4.py \
  --patient_json_dir data/patients_json \
  --db_path data/hpo_genes_with_synonyms.db \
  --hpo_json_path data/hp.json \
  --output_dir out/mapped \
  --top_k 10 \
  --min_sim 0.70 \
  --max_workers 8 \
  --embed_model nomic-embed-text \
  --chat_model llama3.3:70b-instruct-fp16

Input patient JSON format
-------------------------
{
  "subject_id": "P001",
  "data": [
    {"finding": "abdominal pain", "anatomical_region": "abdomen"},
    {"finding": "diarrhoea", "anatomical_region": "gastrointestinal tract"}
  ]
}

DB schema expectations
----------------------
SQLite DB must contain:
- hpo_synonym_embeddings(hpo_id, hpo_name, term, embedding) where embedding is JSON list of floats
- hpo_gene(hpo_id, genes) where genes is a comma+space separated string ("GENE1, GENE2, ...")

Notes
-----
- The LLM is asked to return structured JSON with candidate indices and reasoning.
- If parsing fails, the script falls back to the top-1 by similarity.
- Duplicate HPO terms are removed, keeping only the entry with the highest similarity score.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import numpy as np
from numpy.linalg import norm
import ollama
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Pydantic model for structured LLM output
# -----------------------------------------------------------------------------

class HPOSelectionResponse(BaseModel):
    """Schema for LLM HPO term selection response."""
    selected_indices: List[int] = Field(
        description="List of candidate numbers (1-indexed) that best match the patient finding"
    )
    reasoning: str = Field(
        description="Brief explanation for why these HPO terms were selected"
    )


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(norm(a) * norm(b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def load_hpo_embeddings(db_path: str) -> List[Tuple[str, str, str, np.ndarray]]:
    """Load synonym-enhanced HPO embeddings into memory."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT hpo_id, hpo_name, term, embedding FROM hpo_synonym_embeddings")
    rows = cur.fetchall()
    conn.close()

    out = []
    for hpo_id, hpo_name, term, embedding_str in rows:
        out.append((hpo_id, hpo_name, term, np.array(json.loads(embedding_str), dtype=float)))
    return out


def load_hpo_gene_map(db_path: str) -> Dict[str, List[str]]:
    """Load HPO -> gene list mapping into memory."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT hpo_id, genes FROM hpo_gene")
    rows = cur.fetchall()
    conn.close()

    gene_map: Dict[str, List[str]] = {}
    for hpo_id, genes in rows:
        if genes:
            gene_map[hpo_id] = genes.split(", ")
        else:
            gene_map[hpo_id] = []
    return gene_map


def load_hpo_definitions(hpo_json_path: str) -> Dict[str, str]:
    """
    Load HPO term definitions from an HPO JSON file.

    Expected shape (as in hp.json / hp.owl.json export):
      hpo_json["graphs"][0]["nodes"] list of nodes where
      - CLASS nodes store id and meta.definition.val
    """
    with open(hpo_json_path, "r", encoding="utf-8") as f:
        hpo_json = json.load(f)

    definitions: Dict[str, str] = {}
    graphs = hpo_json.get("graphs", [])
    if not graphs:
        logging.warning("HPO JSON has no 'graphs' field or it's empty.")
        return definitions

    nodes = graphs[0].get("nodes", [])
    for node in nodes:
        node_type = node.get("type")
        if node_type == "CLASS":
            raw_id = node.get("id", "")
            # Often looks like: http://purl.obolibrary.org/obo/HP_0001250
            hpo_id = raw_id.split("/")[-1].replace("_", ":")
            definition = node.get("meta", {}).get("definition", {}).get("val", "") or ""
            definitions[hpo_id] = definition
        elif node_type is None:
            # Some nodes may omit 'type' depending on export; log but continue
            logging.debug(f"Node missing 'type': {node}")
    return definitions


def find_top_hpo_matches(
    finding: str,
    region: str,
    hpo_embeddings: List[Tuple[str, str, str, np.ndarray]],
    hpo_definitions: Dict[str, str],
    embed_model: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_text = f"{finding} in {region}"
    query_embedding = np.array(
        ollama.embeddings(model=embed_model, prompt=query_text)["embedding"],
        dtype=float,
    )

    scored = []
    for hpo_id, hpo_name, matched_term, hpo_embedding in hpo_embeddings:
        sim = cosine_sim(query_embedding, hpo_embedding)
        scored.append((sim, hpo_id, hpo_name, matched_term))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    return [
        {
            "hpo_id": hpo_id,
            "hpo_term": hpo_name,
            "matched_term": matched_term,
            "definition": hpo_definitions.get(hpo_id, "No definition available"),
            "similarity": float(sim),
        }
        for sim, hpo_id, hpo_name, matched_term in top
    ]


def llm_select_best_hpo(
    finding: str,
    region: str,
    candidates: List[Dict[str, Any]],
    chat_model: str,
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to select best candidate indices using structured output.
    If parsing fails, return [].
    """
    # Build the candidates list as a formatted string
    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        definition = c['definition'] if c['definition'] else "No definition available"
        candidates_text += (
            f"{i}. HPO ID: {c['hpo_id']}\n"
            f"   Term: {c['hpo_term']}\n"
            f"   Matched synonym: {c['matched_term']}\n"
            f"   Definition: {definition}\n"
            f"   Similarity score: {c['similarity']:.3f}\n\n"
        )

    prompt = f"""You are a clinical phenotyping expert mapping patient findings to Human Phenotype Ontology (HPO) terms.

## Task
Select the HPO term(s) that best represent the following patient observation:
- **Finding**: "{finding}"
- **Anatomical region**: "{region}"

## Candidate HPO Terms
{candidates_text}

## Selection Criteria
1. **Semantic match**: The HPO term should accurately capture the clinical meaning of the finding
2. **Specificity**: Prefer more specific terms over general ones when the finding supports specificity
3. **Anatomical consistency**: The HPO term should be compatible with the anatomical region specified
4. **Definition alignment**: The term's definition should align with the clinical context

## Instructions
- Select 1-3 candidates that best match the finding (usually just 1 is appropriate)
- Only select multiple terms if they represent distinct, non-overlapping aspects of the finding
- Do NOT select a term just because it has a high similarity score if it doesn't semantically match
- Provide your selected candidate number(s) and brief reasoning

Return your response as JSON with the selected candidate numbers (1-indexed)."""

    try:
        resp = ollama.chat(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            format=HPOSelectionResponse.model_json_schema(),
            options={"temperature": 0},  # More deterministic output
        )
        
        content = resp.get("message", {}).get("content", "")
        
        # Parse and validate the response using Pydantic
        selection = HPOSelectionResponse.model_validate_json(content)
        
        # Log the reasoning for debugging/auditing
        logging.debug(f"LLM reasoning for '{finding}' in '{region}': {selection.reasoning}")
        
        # Extract selected candidates
        selected = []
        for idx in selection.selected_indices:
            if 1 <= idx <= len(candidates):
                selected.append(candidates[idx - 1])
            else:
                logging.warning(f"LLM returned out-of-range index {idx} for {len(candidates)} candidates")
        
        return selected

    except Exception as e:
        logging.warning(f"LLM selection failed for '{finding}' in '{region}': {e}")
        return []


def map_finding_to_hpo(
    finding: str,
    region: str,
    hpo_embeddings,
    hpo_gene_map,
    hpo_definitions,
    embed_model: str,
    chat_model: str,
    top_k: int,
    min_sim: float,
) -> List[Dict[str, Any]]:
    candidates = find_top_hpo_matches(
        finding=finding,
        region=region,
        hpo_embeddings=hpo_embeddings,
        hpo_definitions=hpo_definitions,
        embed_model=embed_model,
        top_k=top_k,
    )

    if not candidates or candidates[0]["similarity"] < min_sim:
        return [{
            "hpo_id": "NA",
            "hpo_term": "NA",
            "matched_term": "NA",
            "genes": [],
            "similarity": "",
        }]

    selected = llm_select_best_hpo(
        finding=finding,
        region=region,
        candidates=candidates,
        chat_model=chat_model,
    )

    # Fallback: if LLM returns nothing parsable, use top-1 similarity
    if not selected:
        selected = [candidates[0]]

    results = []
    for match in selected:
        hpo_id = match["hpo_id"]
        genes = hpo_gene_map.get(hpo_id, [])
        results.append({
            "hpo_id": hpo_id,
            "hpo_term": match["hpo_term"],
            "matched_term": match["matched_term"],
            "genes": genes,
            "similarity": match.get("similarity", ""),
        })
    return results


def deduplicate_hpo_results(mapped_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate HPO terms, keeping only the entry with the highest similarity score
    for each unique HPO ID.
    
    Args:
        mapped_rows: List of mapping result dictionaries
        
    Returns:
        Deduplicated list with unique HPO IDs, preserving the highest similarity entry
    """
    # Dictionary to track best entry for each HPO ID
    best_by_hpo_id: Dict[str, Dict[str, Any]] = {}
    
    for row in mapped_rows:
        hpo_id = row["hpo_id"]
        
        # Skip NA entries from deduplication logic but keep them
        if hpo_id == "NA":
            # Use a unique key for NA entries to preserve them all
            na_key = f"NA_{row['finding']}_{row['region']}"
            best_by_hpo_id[na_key] = row
            continue
        
        # Get similarity score, handling empty string case
        current_sim = row.get("similarity", "")
        if current_sim == "":
            current_sim = -1.0
        else:
            current_sim = float(current_sim)
        
        if hpo_id not in best_by_hpo_id:
            # First time seeing this HPO ID
            best_by_hpo_id[hpo_id] = row
        else:
            # Compare similarity scores and keep the higher one
            existing_sim = best_by_hpo_id[hpo_id].get("similarity", "")
            if existing_sim == "":
                existing_sim = -1.0
            else:
                existing_sim = float(existing_sim)
            
            if current_sim > existing_sim:
                best_by_hpo_id[hpo_id] = row
    
    # Convert back to list, sorted by similarity (highest first)
    deduplicated = list(best_by_hpo_id.values())
    
    # Sort by similarity score descending (NA entries will be at the end)
    def sort_key(r):
        sim = r.get("similarity", "")
        if sim == "" or r["hpo_id"] == "NA":
            return -2.0  # Put NA entries at the end
        return -float(sim)  # Negative for descending order
    
    deduplicated.sort(key=sort_key)
    
    return deduplicated


def process_patient_file(
    patient_file: str,
    output_dir: str,
    hpo_embeddings,
    hpo_gene_map,
    hpo_definitions,
    embed_model: str,
    chat_model: str,
    top_k: int,
    min_sim: float,
) -> str:
    with open(patient_file, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    subject_id = patient_data.get("subject_id") or os.path.splitext(os.path.basename(patient_file))[0]
    logging.info(f"Processing patient: {subject_id}")

    mapped_rows = []
    for entry in patient_data.get("data", []):
        finding = entry.get("finding", "Unknown")
        region = entry.get("anatomical_region", "Unknown")

        matches = map_finding_to_hpo(
            finding=finding,
            region=region,
            hpo_embeddings=hpo_embeddings,
            hpo_gene_map=hpo_gene_map,
            hpo_definitions=hpo_definitions,
            embed_model=embed_model,
            chat_model=chat_model,
            top_k=top_k,
            min_sim=min_sim,
        )

        for m in matches:
            mapped_rows.append({
                "finding": finding,
                "region": region,
                "hpo_id": m["hpo_id"],
                "hpo_term": m["hpo_term"],
                "matched_term": m["matched_term"],
                "genes": m["genes"],
                "similarity": m.get("similarity", ""),
            })

    # Deduplicate HPO terms, keeping highest similarity for each unique HPO ID
    original_count = len(mapped_rows)
    mapped_rows = deduplicate_hpo_results(mapped_rows)
    dedup_count = len(mapped_rows)
    
    if original_count != dedup_count:
        logging.info(f"Patient {subject_id}: Deduplicated {original_count} -> {dedup_count} HPO terms")

    out_csv = os.path.join(output_dir, f"{subject_id}_hpo_mapped.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["finding", "region", "hpo_id", "hpo_term", "matched_term", "genes", "similarity"])
        for r in mapped_rows:
            w.writerow([
                r["finding"],
                r["region"],
                r["hpo_id"],
                r["hpo_term"],
                r["matched_term"],
                ", ".join(r["genes"]),
                r["similarity"],
            ])

    logging.info(f"Saved: {out_csv}")
    return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HPO mapping with embedding retrieval + LLM reranking (Ollama).")
    p.add_argument("--patient_json_dir", required=True, help="Folder containing patient JSON files.")
    p.add_argument("--db_path", required=True, help="Path to SQLite DB with synonym embeddings + gene map (download from HF).")
    p.add_argument("--hpo_json_path", required=True, help="Path to HPO JSON containing definitions (download from HF).")
    p.add_argument("--output_dir", required=True, help="Output directory for per-patient CSV files.")
    p.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model name.")
    p.add_argument("--chat_model", default="llama3.3:70b-instruct-fp16", help="Ollama chat model for candidate selection.")
    p.add_argument("--top_k", type=int, default=10, help="Number of embedding candidates to pass to the LLM.")
    p.add_argument("--min_sim", type=float, default=0.70, help="Minimum similarity to accept any mapping (else NA).")
    p.add_argument("--max_workers", type=int, default=8, help="Threads for parallel patient processing.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("Loading DB tables into memory...")
    hpo_embeddings = load_hpo_embeddings(args.db_path)
    hpo_gene_map = load_hpo_gene_map(args.db_path)
    logging.info(f"Loaded {len(hpo_embeddings):,} embeddings and {len(hpo_gene_map):,} HPO->gene mappings.")

    logging.info("Loading HPO definitions JSON...")
    hpo_definitions = load_hpo_definitions(args.hpo_json_path)
    logging.info(f"Loaded {len(hpo_definitions):,} HPO definitions.")

    json_files = [
        os.path.join(args.patient_json_dir, f)
        for f in os.listdir(args.patient_json_dir)
        if f.lower().endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {args.patient_json_dir}")

    logging.info(f"Found {len(json_files)} patient files. Processing with {args.max_workers} workers...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                process_patient_file,
                patient_file,
                args.output_dir,
                hpo_embeddings,
                hpo_gene_map,
                hpo_definitions,
                args.embed_model,
                args.chat_model,
                args.top_k,
                args.min_sim,
            )
            for patient_file in json_files
        ]
        for fut in as_completed(futures):
            _ = fut.result()

    logging.info("Done.")


if __name__ == "__main__":
    main()

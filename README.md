HPO Mapper: Semantic Phenotype-to-Ontology Mapping Toolkit
<p align="center"> <img width="250" height="250" src="https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/llamas.png"> </p>
Overview

HPO Mapper is a standalone, scalable, AI-assisted toolkit for mapping clinical phenotypic descriptions to the Human Phenotype Ontology (HPO) and associated genes. It ingests structured or semi-structured clinical findings—optionally paired with anatomical regions—and converts them into standardised HPO terms using embedding-based semantic similarity, with optional large language model (LLM) mediation.

HPO Mapper is designed to support robust phenotype normalisation from heterogeneous clinical inputs, enabling downstream genotype–phenotype integration, cohort harmonisation, and precision medicine workflows. All protocols are designed for local execution, supporting privacy-preserving deployment in clinical and research environments.

HPO Mapper Protocols

HPO Mapper is available in three complementary protocols, reflecting increasing levels of AI mediation and flexibility:

Protocol 1 — Embedding-Based Mapping (Published)

📄 Featured in MED
🔗 https://doi.org/10.1016/j.medj.2025.100895

Cosine similarity–based retrieval over an HPO synonym embedding space

Deterministic, explainable mapping

Optimised for scalability and reproducibility

No LLM dependency

Recommended for: large-scale cohort processing and benchmarking

Protocol 2 — Embedding Retrieval + LLM Quality Control

🧠 Cosine similarity + LLM QC

Embedding-based candidate retrieval

LLM performs post-hoc validation of selected HPO terms

Flags potentially incorrect mappings without altering retrieval

Recommended for: high-precision clinical pipelines and audit-aware workflows

Protocol 3 — Embedding Retrieval + LLM-Based HPO Selection

🧠🧠 Cosine similarity + LLM-mediated selection

Top-K candidate HPO terms retrieved via embeddings

LLM selects the most appropriate term(s) using ontology definitions

Supports one-to-many mappings where clinically appropriate

Recommended for: complex phenotypes, ambiguous findings, and research exploration

Workflow Overview
<p align="center"> <img src="HPO Mapper Visual Abstract.jpeg"> </p>
Tool Description

HPO Mapper implements a modular phenotype-mapping pipeline that combines:

Semantic embeddings of HPO terms and synonyms

Cosine similarity–based candidate retrieval

Optional LLM-based quality control or candidate selection

Gene association lookup via curated HPO gene annotations

The tool supports both:

Paired finding + anatomical region inputs

Finding-only inputs when region data are unavailable

HPO Mapper Workflow

Load Clinical Findings
Input findings are provided as structured JSON records.

Candidate Retrieval
Clinical text is embedded and compared against the HPO embedding space using cosine similarity.

LLM Mediation (Optional)

Protocol 2: LLM validates retrieved mappings

Protocol 3: LLM selects best candidate(s) using HPO definitions

Gene Association Mapping
Mapped HPO terms are linked to associated genes using curated HPO gene annotations.

Output Generation
Results are written to CSV with logging for auditing and reproducibility.

<p align="center"> <img src="https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/hpo.png"> </p>
Live Demo and Embeddings

An executable demo and the precomputed HPO embedding space are available via Hugging Face:

🔗 HPO Mapper Demo & Embeddings
👉 https://huggingface.co/spaces/UoS-HGIG/HPOmapper

This space provides:

Interactive demonstration of HPO mapping

Precomputed HPO term and synonym embeddings

Reference implementations of HPO Mapper protocols

Requirements

Install required dependencies:

pip install numpy ollama sqlite3


Pull the embedding model:

ollama pull nomic-embed-text


Precomputed embeddings and reference resources are available via:
👉 https://huggingface.co/spaces/UoS-HGIG/HPOmapper

Input Format

Each input file should be a JSON object containing findings and optional anatomical regions:

{
  "subject_id": "",
  "data": [
    {
      "finding": "",
      "anatomical_region": ""
    }
  ]
}


Template:
👉 https://github.com/UoS-HGIG/IBD_LLM/blob/main/src/HPO_mapper/hpo_input_template.json

Customisation
Similarity Thresholds

Adjust cosine similarity cutoffs to control mapping strictness.

LLM Integration

Enable or disable LLM-based QC or selection depending on performance and compute constraints.

Ontology Updates

Replace or extend the HPO embedding space as new ontology releases become available.

Reference

HPO Mapper: AI-assisted semantic mapping of clinical phenotypes to the Human Phenotype Ontology
Alex Z Kadhim, Zachary Green, Iman Nazari, Jonathan Baker, Michael George, Ashley Heinson,
Matt Stammers, Christopher Kipps, R Mark Beattie, James J Ashton, Sarah Ennis

📄 MED (2025)
🔗 https://doi.org/10.1016/j.medj.2025.100895

License

This project is licensed under the MIT License.

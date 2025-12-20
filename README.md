HPO Mapper: Semantic Phenotype-to-Ontology Mapping Toolkit
<p align="center"> <img width="250" height="250" src="https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/llamas.png"> </p>
Overview

HPO Mapper is a standalone, scalable AI-assisted tool for mapping clinical phenotypic descriptions to the Human Phenotype Ontology (HPO) and associated genes. It ingests semantically structured or semi-structured clinical findings—optionally paired with anatomical regions—and converts them into standardised HPO terms using embedding-based semantic similarity with optional large language model (LLM) mediation.

HPO Mapper supports robust phenotype normalisation from heterogeneous clinical inputs and enables downstream genotype–phenotype integration, cohort harmonisation, and precision medicine workflows. The tool is designed for real-world clinical and research deployment, with support for local execution and privacy-preserving operation.

<p align="center"> <img src="https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/abstract.png"> </p>
Live Demo and Embeddings

An executable demo and the precomputed HPO embedding space are available via the Hugging Face Space:

🔗 HPO Mapper Demo & Embeddings
👉 https://huggingface.co/spaces/UoS-HGIG/HPOmapper

This space provides:

Interactive demonstration of HPO mapping

Precomputed HPO term and synonym embeddings

Reference implementation of the HPO Mapper workflow

Reference

HPO Mapper: AI-assisted semantic mapping of clinical phenotypes to the Human Phenotype Ontology
Alex Z Kadhim, Zachary Green, Iman Nazari, Jonathan Baker, Michael George, Ashley Heinson, Matt Stammers, Christopher Kipps, R Mark Beattie, James J Ashton, Sarah Ennis
Manuscript under review / preprint forthcoming

Tool Description

HPO Mapper implements a modular phenotype-mapping pipeline that combines:

Semantic embeddings of HPO terms and synonyms

Cosine similarity–based candidate retrieval

Optional LLM-based quality control and mediation

Gene association lookup via curated HPO gene annotations

The tool supports both:

Paired finding + anatomical region inputs

Finding-only inputs when region information is unavailable

HPO Mapper Workflow

Load Clinical Findings
Input findings are provided as structured JSON records.

Candidate Retrieval
Clinical text is embedded and compared against the HPO embedding space using cosine similarity.

LLM Mediation (Optional)
An LLM can be used to validate or refine candidate term selection.

Gene Association Mapping
Mapped HPO terms are linked to associated genes using curated HPO gene annotations.

Output Generation
Results are written to CSV with optional logging for auditing and reproducibility.

<p align="center"> <img src="https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/hpo.png"> </p>
Requirements

Install required dependencies:

pip install numpy ollama sqlite3


Pull the embedding model:

ollama pull nomic-embed-text


HPO embeddings and example resources are available via:
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
https://github.com/UoS-HGIG/IBD_LLM/blob/main/src/HPO_mapper/hpo_input_template.json

Customisation

Similarity Thresholds
Adjust cosine similarity cutoffs to control mapping strictness.

LLM Integration
Enable or disable LLM-based quality control depending on performance and compute constraints.

Ontology Updates
Replace or extend the HPO embedding space as new ontology releases become available.

License

This project is licensed under the MIT License.

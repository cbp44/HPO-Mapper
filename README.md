# HPO Mapper: Semantic Phenotype-to-Ontology Mapping Toolkit

![HPO Mapper Workflow](HPO%20Mapper%20Visual%20Abstract.jpeg)

**HPO Mapper** is a standalone, scalable, AI-assisted toolkit for mapping clinical phenotypic descriptions to the **Human Phenotype Ontology (HPO)** and associated genes. It ingests structured or semi-structured clinical findings—optionally paired with anatomical regions—and converts them into standardised HPO terms using **embedding-based semantic similarity**, with optional **large language model (LLM)** mediation.

HPO Mapper supports robust phenotype normalisation from heterogeneous clinical inputs and enables downstream **genotype–phenotype integration**, **cohort harmonisation**, and **precision medicine** workflows. The tool is designed for real-world clinical and research deployment, with support for local execution and privacy-preserving operation.

---

## HPO Mapper Protocols

HPO Mapper is available in **three complementary protocols**, reflecting increasing levels of AI mediation and flexibility.

### Protocol 1 — Embedding-Based Mapping

- Embedding-based semantic similarity over HPO terms and synonyms
- Deterministic and fully explainable
- No LLM dependency
- Designed for scalability and reproducibility

This protocol is described in the following publication:

**Application of generative artificial intelligence to utilize unstructured clinical data for acceleration of inflammatory bowel disease research**  
Kadhim AZ, Green Z, Nazari I, Baker J, George M, Heinson A, Vadgama B, Stammers M, Kipps C, Beattie RM, Ashton JJ, Ennis S  
*MED* (2025)  
https://doi.org/10.1016/j.medj.2025.100895

---

### Protocol 2 — Embedding Retrieval + LLM Quality Control

- Cosine similarity–based candidate retrieval
- LLM performs post-hoc quality control of selected HPO terms
- Incorrect mappings are flagged but not automatically changed

**Recommended for:** high-precision clinical pipelines and audit-aware workflows

---

### Protocol 3 — Embedding Retrieval + LLM-Based HPO Selection

- Cosine similarity–based top-K candidate retrieval
- LLM selects the most appropriate HPO term(s) using ontology definitions
- Supports one-to-many mappings where clinically appropriate

**Recommended for:** complex phenotypes, ambiguous findings, and research exploration

---

## Tool Description

HPO Mapper implements a modular phenotype-mapping pipeline that combines:

- Semantic embeddings of HPO terms and synonyms
- Cosine similarity–based candidate retrieval
- Optional LLM-based quality control or mediation
- Gene association lookup via curated HPO gene annotations

The tool supports both:

- Paired finding + anatomical region inputs
- Finding-only inputs when region information is unavailable

---

## HPO Mapper Workflow

1. **Load Clinical Findings**  
   Input findings are provided as structured JSON records.

2. **Candidate Retrieval**  
   Clinical text is embedded and compared against the HPO embedding space using cosine similarity.

3. **LLM Mediation (Optional)**  
   - Protocol 2: LLM validates retrieved mappings  
   - Protocol 3: LLM selects the best candidate term(s)

4. **Gene Association Mapping**  
   Mapped HPO terms are linked to associated genes using curated HPO gene annotations.

5. **Output Generation**  
   Results are written to CSV with logging for auditing and reproducibility.

![HPO Ontology](https://github.com/UoS-HGIG/IBD_LLM/blob/main/img/hpo.png)

---

## Live Demo and Embeddings

An executable demo and the precomputed HPO embedding space are available via Hugging Face:

**HPO Mapper Demo & Embeddings**  
https://huggingface.co/spaces/UoS-HGIG/HPOmapper

This space provides:

- Interactive demonstration of HPO mapping
- Precomputed HPO term and synonym embeddings
- Reference implementations of HPO Mapper workflows

---

## Requirements

Install required dependencies:

```bash
pip install numpy ollama sqlite3
```
Pull the embedding model:
```
ollama pull nomic-embed-text
```
HPO embeddings and example resources are available via:
https://huggingface.co/spaces/UoS-HGIG/HPOmapper
Input Format

Each input file should be a JSON object containing findings and optional anatomical regions:
```
{
  "subject_id": "",
  "data": [
    {
      "finding": "",
      "anatomical_region": ""
    }
  ]
}
```
### Template:
https://github.com/UoS-HGIG/IBD_LLM/blob/main/src/HPO_mapper/hpo_input_template.json

### License

This project is licensed under the MIT License.

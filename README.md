# Information Retrieval Final Project 2025

The goal was to build an effective reranking system for patent documents using dense retrieval and transformer-based models, evaluated via [Codabench](https://www.codabench.org/).

---

## Objective

To implement and evaluate a two-stage retrieval pipeline:
1. **Candidate Generation** using a dense retriever
2. **Reranking** using fine-tuned transformer-based cross-encoders (e.g., BERT, RoBERTa)

We tackled **Task 2** of the patent matching challenge with a focus on maximizing **Recall@100** and **MAP@100**.

---

## Project Files
 
[`task2_patent_reranking.ipynb`](./notebooks/Task2_Patent_Reranking.ipynb) – Re-ranking using transformer-based cross-encoder  
[`cross_encoder_reranking_train.py`](./src/cross_encoder_reranking_train.py) – Training script for reranker
[`ir_final_presentation.pdf`](./ir_final_presentation.pdf) – Final project slides  
[`submission/`](./submission) – Codabench submission files (`predictions1.json`, `predictions2.json`)

---

## Dataset

The dataset was provided as part of the Codabench Patent Matching Challenge. It includes:

- **Queries**: Based on claims and TAC (Title-Abstract-Claims)
- **Documents**: Scientific publication fields like titles, abstracts, claims, and descriptions
- **Pre-ranking**: Each query is associated with 20 pre-ranked candidate documents

> Due to size and license constraints, the full JSON files (`Citation_JSONs/`, `Content_JSONs/`) are not included.  
To reproduce results, download them from the official challenge and place in a local `datasets/` folder.

---

## Methods & Models

### Task 1: Dense Retriever
- Models used: `all-MiniLM-L6-v2`, `PatentSBERTa`
- Embedding comparison via FAISS cosine similarity
- Used Reciprocal Rank Fusion (RRF) to combine multiple retrievers

### Task 2: Cross-Encoder Reranker
- Models: `infly/inf-retriever-v1-1.5`, `BAAI/bge-reranker-large`
- Fine-tuned with binary relevance on top candidates
- Loss: Binary cross-entropy; Optimizer: AdamW

---

## Evaluation Results

| Metric       | Score   |
|--------------|---------|
| Recall@3     | 0.1283  |
| Recall@5     | 0.2413  |
| Recall@10    | 0.5321  |
| Recall@20    | 0.8627  |
| MAP@100      | 0.2681  |
| MRR          | 0.3878  |
| Mean Rank    | 4.90    |

---

## Technologies Used

- Python, Jupyter Notebook
- Hugging Face Transformers
- FAISS
- PyTorch
- Scikit-learn, Pandas, NumPy
- Codabench Evaluation Platform

---

## Final Presentation

📄 [ir_final_presentation.pdf](./ir_final_presentation.pdf)

---

## Submission Files

| File | Description |
|------|-------------|
| `predictions1.json` | Dense retriever output |
| `predictions2.json` | Transformer reranker output |

You can find them inside the [`submission/`](./submission) folder.

---

## Example Architecture Diagram *(Optional)*

> Insert a visual here if you’d like to show the dual-stage retrieval pipeline

---

## Team Members

- **Md Naim Hassan Saykat** – Task 2: Transformer-based reranking, training, evaluation 
- **Dang Hoang Khang Nguyen** – Task 1: TF-IDF, BM25, Dense retrieval, RRF  
- **Ahmed Nazar** – Contributed to the design, structuring, and visual formatting of the final project presentation  

---

## Disclaimer

This repository is shared for academic and demonstration purposes only.  
Reuse, redistribution, or modification without permission is not allowed.

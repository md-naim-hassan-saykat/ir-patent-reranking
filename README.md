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

[`dense_retrieval_2_0.ipynb`](./dense_retrieval_2_0.ipynb) – Dense retriever for initial candidate retrieval  
[`Task2_Patent_Reranking_...ipynb`](./Task2_Patent_Reranking_...ipynb) – Re-ranking using transformer-based cross-encoder  
[`create_embeddings.py`](./create_embeddings.py) – Script for embedding generation  
[`IR_Presentation.pdf`](./IR_Presentation.pdf) – Final project slides  
[`submission/`](./submission) – Codabench submission files (`predictions1.json`, `predictions2.json`)

---

## Methods & Models

### Dense Retriever
- Dual-encoder setup using BGE, MPNet, and GTR models
- Embeddings generated using HuggingFace models
- FAISS for similarity search

### Cross-Encoder Re-ranker
- Models tested:
  - `BAAI/bge-reranker-large`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained using contrastive loss on query-positive-negative triplets
- Evaluation: `Recall@k`, `MAP@100`, `Mean Rank`

---

## Evaluation Results

| Model | Recall@100 | MAP@100 | Mean Rank |
|-------|------------|---------|------------|
| BGE Dense Retriever | 0.873 | 0.697 | 26.4 |
| Cross-Encoder (BGE-Reranker) | **0.905** | **0.741** | **18.7** |

> Ensemble reranking improved ranking performance by 2.3% over the baseline model.

---

## Technologies Used

- Python, Jupyter Notebook
- Hugging Face Transformers
- FAISS
- PyTorch
- Scikit-learn, Pandas, NumPy
- Codabench Evaluation Platform

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

- **Md Naim Hassan Saykat** – Dense retrieval, reranking pipeline, evaluation, presentation  
- *(add more if applicable)*

---

## Disclaimer

This repository is shared for academic and demonstration purposes only.  
Reuse, redistribution, or modification without permission is not allowed.

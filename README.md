# Patent Reranking with Dense & Cross Encoders
This project implements a patent reranking pipeline using a combination of dense retrieval and cross-encoder reranking models.
The pipeline improves initial retrieval results from TF–IDF and dense embeddings (BGE) with a transformer-based cross-encoder that re-scores query–document pairs.

Evaluation is performed using standard IR metrics such as Mean Average Precision (MAP), Recall@k, and Mean Rank.
## Overview
	•	Dataset: Provided JSONs (train_queries.json, test_queries.json, documents_features.json, train_gold_mapping.json).
	•	Retrievers:
	•	TF–IDF (baseline)
	•	Dense retriever (BGE embeddings)
	•	Re-ranker:
	•	Cross-Encoder (BERT-based pairwise model)
	•	Ensemble:
	•	Reciprocal Rank Fusion (RRF) combining dense retriever + cross-encoder.
	•	Evaluation Metrics: MAP, Recall@k, Mean Rank.
## Repository Structure
ir-patent-reranking/
├── notebooks/
│   └── patent_reranking.ipynb         # Main notebook (end-to-end pipeline)
│
├── src/
│   ├── cross_encode_ranking_train.py  # Cross-encoder training
│   ├── evaluate_train_rankings.py     # Evaluation script
│   └── metrics.py                     # Metric implementations
│
├── data/
│   ├── train_queries.json
│   ├── test_queries.json
│   ├── train_gold_mapping.json
│   └── documents_features.json
│
├── results/
│   ├── predictions_bge_claims.json
│   ├── predictions_tfidf_retriever.json
│   ├── evaluation_metrics.txt
│   └── metrics_result.txt
│
├── docs/
│   ├── report.tex        # LaTeX project report
│   ├── report.pdf        # Compiled report
│   └── references.bib    # References
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
# Getting Started
## Clone the repository
git clone https://github.com/md-naim-hassan-saykat/patent-reranking.git
cd patent-reranking
## Install dependencies
pip install -r requirements.txt
## Run experiments
	•	Dense retrieval baseline
python src/evaluate_train_rankings.py --method dense
 	•	Cross-encoder re-ranking
python src/cross_encode_ranking_train.py --epochs 3 --batch_size 16
  	•	Evaluate metrics
python src/metrics.py --input results/predictions_bge_claims.json
## Results  

| Model                     | MAP   | Recall@10 | Mean Rank |
|----------------------------|-------|-----------|-----------|
| TF–IDF Baseline            | 0.xx  | 0.xx      | xxx       |
| Dense Retriever (BGE)      | 0.xx  | 0.xx      | xxx       |
| Cross-Encoder Re-ranker    | 0.xx  | 0.xx      | xxx       |
| Ensemble (Dense + CE, RRF) | 0.xx  | 0.xx      | xxx       |

## References
	•	Lin et al., Dense Passage Retrieval for Open-Domain Question Answering, ACL 2020.
	•	Nogueira & Cho, Passage Re-ranking with BERT, arXiv 2019.
	•	Hugging Face Transformers: https://huggingface.co/docs/transformers.

 ## Author

 **Md Naim Hassan Saykat**  
*MSc in Artificial Intelligence, Université Paris-Saclay*  

[LinkedIn](https://www.linkedin.com/in/md-naim-hassan-saykat/)  
[GitHub](https://github.com/md-naim-hassan-saykat)  
[Academic Email](mailto:md-naim-hassan.saykat@universite-paris-saclay.fr)  
[Personal Email](mailto:mdnaimhassansaykat@gmail.com)

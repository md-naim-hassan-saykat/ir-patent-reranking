Patent Reranking with Dense & Cross Encoders
This project implements a patent reranking pipeline using a combination of dense retrieval and cross-encoder reranking models.
The system improves initial retrieval results from TF–IDF and dense embeddings (BGE) with a transformer-based cross-encoder that re-scores query–document pairs.
Evaluation is performed using standard IR metrics such as MAP, Recall@k, and Mean Rank.
Overview
	•	Dataset: JSON files provided (train_queries.json, test_queries.json, documents_features.json, train_gold_mapping.json).
	•	Retrievers:
	•	TF–IDF (baseline)
	•	Dense embeddings (BGE)
	•	Re-ranker:
	•	Cross-Encoder (BERT-based)
	•	Ensemble:
	•	Reciprocal Rank Fusion (RRF) combining dense retrieval + cross-encoder.
	•	Evaluation Metrics: MAP, Recall@k, Mean Rank.

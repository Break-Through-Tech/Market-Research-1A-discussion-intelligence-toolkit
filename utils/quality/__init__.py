"""
Conversation quality scoring prototype (issue #16).

    cga_cmv      CGA-CMV dataset: target, context rules, official pair-preserving splits
    scorers      constant, length, TF-IDF and sentence-embedding scorers
    embeddings   cached sentence-transformer embedder (TextEmbedder protocol)
    metrics      ROC-AUC, Spearman, MAE, Brier, accuracy, pair-bootstrap CIs

Experiments: experiments/quality_scoring/run_quality_experiments.py
Write-up:    docs/quality-scoring.md
"""

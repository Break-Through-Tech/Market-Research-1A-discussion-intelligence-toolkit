\# Issue #17 - Topic Clustering Evaluation



\## Objective



Evaluate whether discussion utterances can be grouped into meaningful topics using TF-IDF and semantic embeddings.



\## Methods



We used 12,000 utterances from the processed discussion dataset.



Two clustering methods were tested:



\- TF-IDF with MiniBatch K-Means, selecting 8 clusters.

\- MiniLM sentence embeddings with K-Means, selecting 16 clusters.



We compared the models using silhouette scores, NPMI topic coherence, and manual inspection of sampled utterances.



NPMI was calculated using document-level word co-occurrence across the 12,000 sampled utterances.



\## Results



| Metric | TF-IDF | Semantic |

|---|---:|---:|

| Selected clusters | 8 | 16 |

| Silhouette score | 0.00642 | 0.03370 |

| Mean NPMI | 0.19646 | 0.17056 |

| Topics flagged for review | 6/8 | 14/16 |



The semantic approach achieved a higher silhouette score, suggesting better relative separation between clusters. However, both silhouette scores remained close to zero.



TF-IDF achieved a slightly higher NPMI score. One of its highest-scoring clusters was dominated by URL-related terms, which demonstrates that keyword co-occurrence does not always represent meaningful topics.



\## Manual Evaluation



Five samples from each of the 16 semantic clusters were inspected, for a total of 80 utterances.



We identified recognizable themes including:



\- Gaming and game strategies

\- Competitive sports

\- Entertainment and media

\- Technology and hardware

\- Shopping and financial decisions

\- Personal experiences and relationships



Several other clusters did not have clear, consistent subjects.



\## Observed Limitations



1\. Generic conversational words appeared frequently in topic keywords.

2\. URL-related terms influenced some clusters.

3\. Some clusters contained unrelated subjects.

4\. Similar topics, particularly gaming, appeared across multiple clusters.

5\. Short replies sometimes lacked enough conversation context for meaningful topic identification.



\## Conclusion



Semantic clustering was more effective at separating discussion utterances according to silhouette scores, but neither approach consistently produced well-defined topics.



Additional preprocessing, conversation-context integration, and clustering experiments could improve topic quality.



The current results provide an initial baseline for future topic discovery work.



\## Reproduction



Run the clustering experiments and evaluation from the repository root:



```bash

python experiments/franceli/topic\_clustering\_baseline.py

python experiments/franceli/semantic\_topic\_clustering.py

python experiments/franceli/topic\_clustering\_evaluation.py

```



Evaluation outputs are stored locally under `artifacts/topic\_clustering/evaluation/` and are excluded from Git by the repository's `.gitignore`.


from collections import Counter
from convokit import Corpus
import matplotlib.pyplot as plt

corpus = Corpus(
    filename=r"C:\Users\User\.convokit\saved-corpora\reddit-coarse-discourse-corpus"
)

counts = Counter(
    u.meta.get("majority_type")
    for u in corpus.iter_utterances()
    if u.meta.get("majority_type") is not None
)

labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, values)

plt.title("Discourse Class Distribution (Pre-Cleaning)")
plt.xlabel("Discourse Type")
plt.ylabel("Number of Utterances")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "discourse_class_distribution_pre_cleaning.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
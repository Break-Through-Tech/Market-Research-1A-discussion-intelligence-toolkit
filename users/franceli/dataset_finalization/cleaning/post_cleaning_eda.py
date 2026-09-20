import csv
from collections import Counter
import matplotlib.pyplot as plt

input_file = (
    r"users\franceli\dataset_finalization\cleaning\cleaned_discourse_dataset.csv"
)

counts = Counter()

with open(input_file, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        counts[row["majority_type"]] += 1

labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, values)

plt.title("Discourse Class Distribution (Post-Cleaning)")
plt.xlabel("Discourse Type")
plt.ylabel("Number of Utterances")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    r"users\franceli\dataset_finalization\cleaning\discourse_class_distribution_post_cleaning.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Post-cleaning class counts:")
print(counts)
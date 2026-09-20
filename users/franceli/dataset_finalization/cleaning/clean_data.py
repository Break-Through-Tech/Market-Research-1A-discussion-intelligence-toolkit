import csv
from convokit import Corpus

corpus = Corpus(
    filename=r"C:\Users\User\.convokit\saved-corpora\reddit-coarse-discourse-corpus"
)

output_file = (
    r"users\franceli\dataset_finalization\cleaning\cleaned_discourse_dataset.csv"
)

total = 0
kept = 0
removed_missing_label = 0
removed_empty = 0
removed_deleted = 0
removed_removed = 0

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "utterance_id",
        "conversation_id",
        "speaker_id",
        "reply_to",
        "text",
        "majority_type",
        "post_depth",
        "ups",
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for u in corpus.iter_utterances():
        total += 1

        text = str(u.text).strip()
        label = u.meta.get("majority_type")

        if label is None:
            removed_missing_label += 1
            continue

        if not text:
            removed_empty += 1
            continue

        if text.lower() == "[deleted]":
            removed_deleted += 1
            continue

        if text.lower() == "[removed]":
            removed_removed += 1
            continue

        writer.writerow(
            {
                "utterance_id": u.id,
                "conversation_id": u.conversation_id,
                "speaker_id": u.speaker.id if u.speaker else None,
                "reply_to": u.reply_to,
                "text": text,
                "majority_type": label,
                "post_depth": u.meta.get("post_depth"),
                "ups": u.meta.get("ups"),
            }
        )

        kept += 1

print("Cleaning complete.")
print("Total utterances:", total)
print("Kept utterances:", kept)
print("Removed - missing label:", removed_missing_label)
print("Removed - empty text:", removed_empty)
print("Removed - [deleted]:", removed_deleted)
print("Removed - [removed]:", removed_removed)
print("Saved cleaned dataset to:", output_file)
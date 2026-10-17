from collections import defaultdict
from convokit import Corpus

corpus = Corpus(
    filename=r"C:\Users\User\.convokit\saved-corpora\reddit-coarse-discourse-corpus"
)

text_groups = defaultdict(list)

for u in corpus.iter_utterances():
    text = str(u.text).strip()

    if text:
        text_groups[text].append(u)

duplicates = {
    text: utterances
    for text, utterances in text_groups.items()
    if len(utterances) > 1
}

print("Unique duplicated texts:", len(duplicates))

extra_rows = sum(len(utterances) - 1 for utterances in duplicates.values())
print("Extra duplicate rows:", extra_rows)

print("\nExample repeated texts:\n")

shown = 0

for text, utterances in sorted(
    duplicates.items(),
    key=lambda item: len(item[1]),
    reverse=True
):
    print("=" * 70)
    print("TEXT:", repr(text[:200]))
    print("OCCURRENCES:", len(utterances))

    for u in utterances[:5]:
        print(
            "ID:", u.id,
            "| Conversation:", u.conversation_id,
            "| Reply to:", u.reply_to,
            "| Label:", u.meta.get("majority_type")
        )

    shown += 1

    if shown == 15:
        break
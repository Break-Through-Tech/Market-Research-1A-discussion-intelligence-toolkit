
from collections import Counter

from utils.modeling_data.build import load_prepared_splits


print("Loading prepared dataset...", flush=True)

splits = load_prepared_splits()


def inspect_split(name, examples):
    total = len(examples)

    with_parent = [
        example for example in examples
        if isinstance(example.metadata.get("parent_text"), str)
        and example.metadata["parent_text"].strip()
    ]

    print(f"\n{name.upper()} SPLIT")
    print(f"Total examples: {total}")
    print(f"Examples with parent text: {len(with_parent)}")
    print(f"Examples without parent text: {total - len(with_parent)}")
    print(f"Parent text availability: {len(with_parent) / total:.2%}")

    print("\nParent context availability by label:")

    label_counts = Counter(example.label for example in examples)
    parent_counts = Counter(example.label for example in with_parent)

    for label in sorted(label_counts):
        available = parent_counts[label]
        total_label = label_counts[label]
        print(f"{label}: {available}/{total_label} ({available / total_label:.2%})")

    if with_parent:
        example = with_parent[0]
        print("\nExample parent comment:")
        print(repr(example.metadata["parent_text"][:300]))
        print("\nExample target comment:")
        print(repr(example.text[:300]))


inspect_split("train", splits.train)
inspect_split("dev", splits.dev)

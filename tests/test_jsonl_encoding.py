import json

from artifacts.corpora import (
    load_jsonl_classification,
    load_jsonl_regression,
)


def test_classification_utf8(tmp_path):
    path = tmp_path / "classification.jsonl"

    example = {
        "id": "test_001",
        "text": "Café prices are rising 📈",
        "label": "negative_reaction",
    }

    path.write_text(
        json.dumps(example, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    examples = load_jsonl_classification(path)

    assert len(examples) == 1
    assert examples[0].text == example["text"]
    assert examples[0].label == "negative_reaction"


def test_regression_utf8(tmp_path):
    path = tmp_path / "regression.jsonl"

    example = {
        "id": "test_002",
        "text": "El precio está subiendo 📈",
        "score": 0.75,
    }

    path.write_text(
        json.dumps(example, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    examples = load_jsonl_regression(path)

    assert len(examples) == 1
    assert examples[0].text == example["text"]

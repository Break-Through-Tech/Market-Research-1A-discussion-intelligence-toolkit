from collections import Counter

import pytest

from artifacts.corpora import ClassificationExample, split_by_group
from utils.modeling_data.splits import assign_split, read_manifest, write_manifest

IDS = [f"t3_{i:05d}" for i in range(5000)]


def test_assignment_is_deterministic():
    assert [assign_split(i) for i in IDS] == [assign_split(i) for i in IDS]


def test_assignment_matches_existing_loader_split():
    """Thread-level splits must equal CoarseDiscourseLoader's, so PR #18 baselines stay comparable."""
    examples = [ClassificationExample(id=i, text="x", label="a", metadata={"cid": i}) for i in IDS]
    loader = split_by_group(examples, lambda ex: ex.metadata["cid"], seed=42)
    for name in ("train", "dev", "test"):
        assert all(assign_split(ex.id) == name for ex in getattr(loader, name))


def test_fractions_are_close_to_target():
    shares = Counter(assign_split(i) for i in IDS)
    assert shares["test"] / len(IDS) == pytest.approx(0.1, abs=0.02)
    assert shares["dev"] / len(IDS) == pytest.approx(0.1, abs=0.02)


def test_seed_changes_assignment():
    assert [assign_split(i, seed=1) for i in IDS[:200]] != [assign_split(i, seed=2) for i in IDS[:200]]


@pytest.mark.parametrize("dev,test", [(0.5, 0.5), (-0.1, 0.1), (0.6, 0.5)])
def test_invalid_fractions_raise(dev, test):
    with pytest.raises(ValueError):
        assign_split("t3_x", dev_frac=dev, test_frac=test)


def test_manifest_roundtrip_is_sorted(tmp_path):
    rows = [{"conversation_id": "t3_b", "subreddit": "x", "split": "dev"},
            {"conversation_id": "t3_a", "subreddit": None, "split": "train"}]
    path = tmp_path / "m.csv"
    write_manifest(rows, path)
    assert path.read_text().splitlines() == [
        "conversation_id,subreddit,split", "t3_a,,train", "t3_b,x,dev"]
    assert read_manifest(path) == {"t3_a": "train", "t3_b": "dev"}

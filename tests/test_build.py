import json

from utils.modeling_data.build import BuildConfig, build, load_prepared_splits, main
from utils.modeling_data.splits import read_manifest

OUTPUTS = ["conversations.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl",
           "split_manifest.csv", "dataset_card.json"]


def _build(corpus, tmp_path, name="out", **kw):
    cfg = BuildConfig(corpus_dir=corpus, out_dir=tmp_path / name, publish_dir=tmp_path / f"{name}_pub", **kw)
    return build(cfg), cfg


def test_end_to_end_build(tiny_corpus, tmp_path):
    result, cfg = _build(tiny_corpus, tmp_path)
    for f in OUTPUTS:
        assert (cfg.out_dir / f).exists(), f
    assert (cfg.publish_dir / "coarse_discourse_split_manifest.csv").exists()
    assert result.report.ok, result.report.format()

    card = json.loads((cfg.out_dir / "dataset_card.json").read_text())
    # 60 threads x 4 labeled utterances + edge thread root + orphan
    assert card["total_examples"] == 60 * 4 + 2
    assert sum(c["examples"] for c in card["counts"].values()) == card["total_examples"]
    assert sum(c["conversations"] for c in card["counts"].values()) == 61
    assert card["exclusions"] == {"deleted_text": 1, "dropped_label": 1,
                                  "empty_text": 1, "no_majority_label": 1}
    assert card["normalization"]["orphans_reparented"] == 1
    assert "negative_reaction" in card["modeling_labels"]


def test_build_is_byte_reproducible(tiny_corpus, tmp_path):
    _, a = _build(tiny_corpus, tmp_path, "a")
    _, b = _build(tiny_corpus, tmp_path, "b")
    for f in OUTPUTS:
        assert (a.out_dir / f).read_bytes() == (b.out_dir / f).read_bytes(), f


def test_prepared_splits_reload(tiny_corpus, tmp_path):
    result, cfg = _build(tiny_corpus, tmp_path)
    loaded = load_prepared_splits(cfg.out_dir)
    assert loaded.summary() == result.splits.summary()
    manifest = read_manifest(cfg.out_dir / "split_manifest.csv")
    for name in ("train", "dev", "test"):
        assert all(manifest[e.metadata["conversation_source_id"]] == name for e in getattr(loaded, name))


def test_subreddit_filter(tiny_corpus, tmp_path):
    result, cfg = _build(tiny_corpus, tmp_path, subreddits=frozenset({"buildapc"}))
    assert set(read_manifest(cfg.out_dir / "split_manifest.csv")) == (
        {f"t3_c{i:03d}" for i in range(60) if i % 2} | {"t3_edge"})
    assert result.card["normalization"]["conversations_filtered_by_subreddit"] == 30
    assert (cfg.publish_dir / "coarse_discourse_subset_split_manifest.csv").exists()


def test_cli_reports_missing_corpus(tmp_path):
    assert main(["--corpus-dir", str(tmp_path / "nope"), "--no-publish"]) == 2


def test_cli_runs(tiny_corpus, tmp_path):
    out = tmp_path / "cli"
    assert main(["--corpus-dir", str(tiny_corpus), "--out-dir", str(out), "--no-publish"]) == 0
    assert (out / "train.jsonl").exists()

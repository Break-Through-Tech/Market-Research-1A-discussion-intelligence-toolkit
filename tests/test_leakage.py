from artifacts.corpora import ClassificationExample, ClassificationSplits
from utils.modeling_data.leakage import normalize_text, run_leakage_checks

LABELS = ["question", "answer", "agreement"]


def ex(id, conv, label="answer", text=None):
    return ClassificationExample(
        id=id, text=text or f"unique text for {id}", label=label,
        metadata={"conversation_id": f"reddit:{conv}", "conversation_source_id": conv})


def clean_splits():
    def block(prefix, n_conv):
        return [ex(f"{prefix}{c}_{l}", f"{prefix}{c}", l) for c in range(n_conv) for l in LABELS]
    train, dev, test = block("tr", 10), block("dv", 2), block("te", 2)
    manifest = {e.metadata["conversation_source_id"]: s
                for s, xs in (("train", train), ("dev", dev), ("test", test)) for e in xs}
    return ClassificationSplits(train=train, dev=dev, test=test), manifest


def status(report, name):
    return next(c.status for c in report.checks if c.name == name)


def test_clean_splits_pass_everything():
    splits, manifest = clean_splits()
    report = run_leakage_checks(splits, manifest)
    assert report.ok
    assert all(c.status == "pass" for c in report.checks), report.format()


def test_thread_split_across_sets_fails():
    splits, manifest = clean_splits()
    splits.test.append(ex("leak", "tr0"))
    report = run_leakage_checks(splits, manifest)
    assert not report.ok
    assert status(report, "conversation_disjoint") == "fail"
    assert status(report, "manifest_consistency") == "fail"


def test_duplicate_example_id_fails():
    splits, manifest = clean_splits()
    splits.train.append(splits.train[0])
    assert status(run_leakage_checks(splits, manifest), "example_disjoint") == "fail"


def test_missing_label_in_eval_split_warns():
    splits, manifest = clean_splits()
    splits.test[:] = [e for e in splits.test if e.label != "agreement"]
    report = run_leakage_checks(splits, manifest)
    assert report.ok
    assert status(report, "label_coverage") == "warn"
    assert status(report, "label_distribution") == "warn"


def test_short_generic_overlap_passes_but_long_overlap_warns():
    splits, manifest = clean_splits()
    splits.train.append(ex("tr0_x", "tr0", text="Thanks!"))
    splits.dev.append(ex("dv0_x", "dv0", text="thanks"))
    report = run_leakage_checks(splits, manifest)
    assert status(report, "text_overlap") == "pass"
    overlap = next(c for c in report.checks if c.name == "text_overlap")
    assert overlap.data["dev"]["overlapping_examples"] == 1

    long = "this exact long paragraph was copy pasted into two different threads"
    splits.train.append(ex("tr1_x", "tr1", text=long))
    splits.test.append(ex("te0_x", "te0", text=long.upper()))
    assert status(run_leakage_checks(splits, manifest), "text_overlap") == "warn"


def test_normalize_text():
    assert normalize_text("  Thanks!!\n") == "thanks"
    assert normalize_text("Yes.  OK") == "yes. ok"

"""Show every section of the report layout on a real reference thread.

No trained model runs here. The discourse labels, themes and civility value
come from the placeholder rules below, so the output illustrates the format in
docs/report-format.md and says nothing about the thread or about model quality.

    uv run --offline python examples/report_format_demo.py > examples/full-report-example.md
"""

from pathlib import Path

from discussion_intelligence import analyze
from discussion_intelligence.reports import ConversationQuality
from discussion_intelligence.sample import load_reference_conversations

SAMPLE = Path(__file__).resolve().parents[1] / "data/coarse-discourse-reference-sample.jsonl"


class PlaceholderClassifier:
    """Keyword rules standing in for the discourse classifier."""

    def predict(self, utterances):
        labels = {}
        for u in utterances:
            text = u.text.lower()
            if "?" in text:
                labels[u.id] = "question"
            elif "agree" in text or text.startswith("yes"):
                labels[u.id] = "agreement"
            elif text.startswith("no") or "but " in text:
                labels[u.id] = "disagreement"
            else:
                labels[u.id] = "answer"
        return labels


class PlaceholderClusterer:
    """Group each reply with the top-level comment it descends from."""

    def assign(self, utterances):
        parents = {u.id: u.parent_id for u in utterances}
        branches = {}
        themes = {}
        for u in utterances:
            node = u.id
            while parents.get(parents.get(node)) is not None:
                node = parents[node]
            if parents.get(node) is None:
                continue  # the original post starts every branch, so it stays unassigned
            branches.setdefault(node, str(len(branches) + 1))
            themes[u.id] = branches[node]
        return themes


class PlaceholderScorer:
    """A fixed value standing in for the civility scorer."""

    def score(self, utterances):
        return ConversationQuality(
            value=0.5,
            evidence_ids=[u.id for u in utterances],
            model_name="placeholder, not a trained model",
        )


def build_report():
    return analyze(
        load_reference_conversations(SAMPLE)[0],
        top_k=5,
        classifier=PlaceholderClassifier(),
        clusterer=PlaceholderClusterer(),
        quality_scorer=PlaceholderScorer(),
    )


if __name__ == "__main__":
    print(build_report().to_markdown())

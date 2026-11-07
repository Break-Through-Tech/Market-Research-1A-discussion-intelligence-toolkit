from datetime import datetime, timezone

import numpy as np
import pytest
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from discussion_intelligence import analyze
from discussion_intelligence.adapters import (
    ContextClassifierAdapter,
    ConversationCivilityAdapter,
    InferenceUtterance,
    TextClassifierAdapter,
)


def test_real_fitted_text_classifier_adapter(conversation):
    texts = ["how why question", "what question", "RAID disk storage", "GPU gaming answer"]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(texts, ["question", "question", "answer", "answer"])
    # Fitting belongs to setup/training, and is forbidden during analyze().
    model.fit = lambda *args, **kwargs: pytest.fail("Inference must not train")
    report = analyze(conversation, classifier=TextClassifierAdapter(model))
    expected = model.predict([u.text for u in conversation.utterances])
    assert [report.evidence[u.id].discourse_act for u in conversation.utterances] == list(expected)


def test_context_adapter_uses_target_then_parent_and_never_fits(conversation):
    targets = ["why question", "what question", "RAID data answer", "gaming answer"]
    parents = ["", "", "data root", "graphics root"]
    target = TfidfVectorizer().fit(targets)
    parent = TfidfVectorizer().fit(parents)
    features = hstack([target.transform(targets), parent.transform(parents)], format="csr")
    model = LogisticRegression().fit(features, ["question", "question", "answer", "answer"])
    seen = []
    original_predict = model.predict

    def predict(values):
        seen.append(values.copy())
        return original_predict(values)

    model.predict = predict
    adapter = ContextClassifierAdapter(model, target, parent)
    report = analyze(conversation, classifier=adapter)
    assert report.components["discourse"].state == "available"
    expected_parent = ["", conversation.root().text, conversation.root().text]
    expected = hstack(
        [
            target.transform([u.text for u in conversation.utterances]),
            parent.transform(expected_parent),
        ],
        format="csr",
    )
    assert np.allclose(seen[0].toarray(), expected.toarray())


def test_classifier_length_mismatch_rejected(conversation):
    class Model:
        def predict(self, texts):
            return ["answer"]

    report = analyze(conversation, classifier=TextClassifierAdapter(Model()))
    assert report.components["discourse"].state == "failed"


def test_civility_adapter_empty_context_rejected():
    with pytest.raises(ValueError):
        ConversationCivilityAdapter(object(), model_name="test").score([])


def test_civility_adapter_uses_context_not_argument_slots():
    class Model:
        def predict(self, contexts):
            assert contexts == [["observed comment"]]
            return [0.42]

    score = ConversationCivilityAdapter(Model(), model_name="fitted").score(
        [InferenceUtterance("reddit:t1_a", "observed comment", None, "", "reddit:u/a")]
    )
    assert score.value == 0.42
    assert score.evidence_ids == ["reddit:t1_a"]


def test_civility_adapter_orders_real_timestamps():
    class Model:
        def predict(self, contexts):
            assert contexts == [["earlier", "later"]]
            return [0.5]

    earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = datetime(2026, 2, 1, tzinfo=timezone.utc)
    context = [
        InferenceUtterance("reddit:t1_b", "later", None, "", "reddit:u/a", later),
        InferenceUtterance("reddit:t1_a", "earlier", None, "", "reddit:u/a", earlier),
    ]
    score = ConversationCivilityAdapter(Model(), model_name="test").score(context)
    assert score.evidence_ids == ["reddit:t1_a", "reddit:t1_b"]

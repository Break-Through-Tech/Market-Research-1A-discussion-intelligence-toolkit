"""Load explicitly trusted fitted artifacts. Joblib/pickle can execute code."""

from collections.abc import Mapping
from pathlib import Path

from .adapters import (
    ContextClassifierAdapter,
    ConversationCivilityAdapter,
    TextClassifierAdapter,
)


def load_trusted_bundle(path: str | Path) -> dict[str, object]:
    """Return analyze() dependencies from the documented model-owner mapping.

    Call only after an explicit request to deserialize this trusted file. Shape
    checks happen AFTER deserialization and do not make an untrusted file safe.
    No fitting, training-experiment imports or model downloads occur here.
    """
    import joblib

    try:
        bundle = joblib.load(path)
    except OSError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not load trusted bundle ({type(exc).__name__}); "
            "check the artifact and install its compatible model code/dependencies."
        ) from exc
    if not isinstance(bundle, Mapping):
        raise ValueError("Trusted bundle must contain a mapping of fitted model objects")
    context_keys = {"context_model", "target_vectorizer", "parent_vectorizer"}
    if context_keys.intersection(bundle) and not context_keys.issubset(bundle):
        raise ValueError(
            "Context bundle requires context_model, target_vectorizer and parent_vectorizer"
        )
    classifier = None
    # Preserve the existing example's precedence when both classifiers are present.
    if "context_model" in bundle:
        classifier = ContextClassifierAdapter(
            bundle["context_model"], bundle["target_vectorizer"], bundle["parent_vectorizer"]
        )
    elif "discourse_model" in bundle:
        classifier = TextClassifierAdapter(bundle["discourse_model"])
    quality = None
    if "civility_model" in bundle:
        quality = ConversationCivilityAdapter(
            bundle["civility_model"], model_name=bundle.get("civility_model_name", "owner-supplied")
        )
    return {
        "classifier": classifier,
        "quality_scorer": quality,
        "clusterer": bundle.get("topic_clusterer"),
        "embedder": bundle.get("embedder"),
    }

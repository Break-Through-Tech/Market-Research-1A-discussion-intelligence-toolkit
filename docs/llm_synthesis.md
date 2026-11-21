
# LLM Synthesis

## Overview

The LLM synthesis component uses OpenAI to identify important
discussion excerpts from retrieved Reddit comments.

Each selected quotation must be supported by an original
comment and linked to its evidence ID.

## Implementation

The implementation is located in:

`src/discussion_intelligence/llm_synthesis.py`

It provides:

- `LLMSynthesisProvider`: Builds prompts, processes model
  responses, and validates evidence citations.
- `create_openai_provider()`: Creates a provider using the
  OpenAI Responses API.

## Example Usage

```python
from discussion_intelligence.llm_synthesis import create_openai_provider
from discussion_intelligence.pipeline import analyze

provider = create_openai_provider(model="gpt-4.1-mini")

report = analyze(
    conversation,
    query="What are the main discussion points?",
    synthesis_provider=provider,
)
```

The `conversation` variable must contain a valid canonical
Conversation object.

## API Configuration

Install project dependencies using:

```bash
uv sync
```

Configure the `OPENAI_API_KEY` environment variable before
making live API requests.

Do not commit API keys to the repository.

OpenAI API access may require billing or project permissions.

## Evidence Validation

The provider:

1. Uses only supplied Reddit evidence.
2. Requests exact quotations from the language model.
3. Checks that referenced evidence IDs exist.
4. Rejects quotations that do not match the source text.
5. Rejects duplicate evidence IDs.

If synthesis fails during `analyze()`, the existing pipeline
uses its deterministic extractive fallback.

The current report schema supports exact excerpts rather
than unrestricted paraphrased summaries.

## Testing

Run synthesis tests:

```bash
uv run pytest tests/test_llm_synthesis.py -v
```

Run the full test suite:

```bash
PYTHONPATH=src:. uv run pytest -q
```

Run code-quality checks:

```bash
uv run ruff check src/discussion_intelligence/llm_synthesis.py tests/test_llm_synthesis.py
```

Tests use mocked model responses and do not require an API key.

## Limitations

- Live OpenAI API behavior has not yet been verified.
- Synthesis is limited to evidence retrieved by the pipeline.
- Output is extractive because the report schema requires
  exact quotations.
- A configured API key and accessible model are required
  for live generation.

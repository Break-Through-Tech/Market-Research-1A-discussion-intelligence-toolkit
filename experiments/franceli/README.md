

\# Discourse Classifier Experiments



\## Overview



This work evaluates machine learning models for classifying

Reddit comments into nine discourse categories as part of the

Market Research 1A Discussion Intelligence Toolkit.



The goal is to determine whether class balancing and parent-comment

context improve discourse classification.



\## Dataset



The experiments use the standardized modeling dataset prepared

by the project's dataset pipeline.



| Split | Examples |

|---|---:|

| Training | 79,010 |

| Development | 9,740 |

| Held-out test | 9,555 |



The nine discourse categories are:



\- agreement

\- announcement

\- answer

\- appreciation

\- disagreement

\- elaboration

\- humor

\- negative\_reaction

\- question



The dataset is split by conversation to reduce leakage between

training, development, and test examples.



\## Model Architecture



The baseline classifier uses:



\- TF-IDF text vectorization

\- Logistic Regression

\- Maximum 10,000 TF-IDF features

\- Maximum 1,000 Logistic Regression iterations



The context-aware classifier uses separate TF-IDF representations

for the target comment and its parent comment, providing up to

20,000 features in total.



The two sparse feature matrices are concatenated before training

Logistic Regression.



Examples without usable parent text receive an empty parent-text

representation.



\## Class Balancing Experiments



We evaluated different levels of class balancing to address

the imbalance between discourse categories.



| Balancing Strength | Accuracy | Macro F1 | Weighted F1 |

|---|---:|---:|---:|

| 0% | 0.5242 | 0.3018 | 0.4804 |

| 25% | 0.5175 | 0.3537 | 0.4890 |

| 50% | 0.4972 | 0.3590 | 0.4849 |

| 100% | 0.3738 | 0.3179 | 0.3991 |



The 50% balanced configuration achieved the highest development

macro F1 among the tested class-weight settings.



Class weights were calculated by interpolating between uniform

weights and scikit-learn's balanced class weights.



\## Error Analysis



The 50% balanced text-only model frequently confused the

following categories:



| Actual Label | Predicted Label | Errors |

|---|---|---:|

| elaboration | answer | 1,023 |

| question | answer | 463 |

| answer | elaboration | 384 |

| answer | question | 306 |



These errors motivated testing parent-comment context.



\## Parent-Comment Availability



Parent-comment text was available for:



\- 65,932 of 79,010 training examples (83.45%)

\- 8,142 of 9,740 development examples (83.59%)



No announcement examples in either split had parent text.



This is a potential dataset-specific structural signal and should

be considered when interpreting improvements.



\## Context Ablation Study



Three configurations were evaluated using identical dataset

splits and 50% class balancing.



| Configuration | Accuracy | Macro F1 | Weighted F1 |

|---|---:|---:|---:|

| Target text only | 0.4972 | 0.3590 | 0.4849 |

| Target text + parent availability | 0.5091 | 0.3887 | 0.5018 |

| Target text + parent text | 0.5565 | 0.4183 | 0.5623 |



Adding parent-text features increased development macro F1

by 0.0593 compared with the text-only model.



The ablation indicates that both structural information and

parent-text features contribute to predictive performance.



\## Per-Class Results



| Category | Text-Only F1 | Context-Aware F1 |

|---|---:|---:|

| agreement | 0.40 | 0.40 |

| announcement | 0.24 | 0.40 |

| answer | 0.62 | 0.69 |

| appreciation | 0.70 | 0.70 |

| disagreement | 0.17 | 0.27 |

| elaboration | 0.22 | 0.41 |

| humor | 0.14 | 0.14 |

| negative\_reaction | 0.21 | 0.19 |

| question | 0.52 | 0.55 |



The largest improvement was observed for elaboration,

increasing from 0.22 to 0.41 F1.



\## Selected Configuration



The development-selected classifier uses:



\- Logistic Regression

\- 50% interpolated class balancing

\- 10,000 TF-IDF features for target text

\- 10,000 TF-IDF features for parent text

\- Separate target and parent vectorizers

\- Conversation-level dataset splits



Development results:



\- Accuracy: 0.5565

\- Macro F1: 0.4183

\- Weighted F1: 0.5623



\## Limitations



\- The evaluation results above are from the development set.

\- The held-out test set has not yet been evaluated.

\- Humor and negative\_reaction remain challenging categories.

\- Parent availability may provide dataset-specific shortcuts.

\- The context-aware classifier uses more features than the

&#x20; text-only baseline.

\- Further evaluation is required to establish generalization

&#x20; beyond this dataset.



\## Reproducibility



Run from the repository root on Windows:



```cmd

uv run --no-project --with pydantic --with scikit-learn --with numpy python -m experiments.franceli.context\_ablation

```



For per-class development evaluation:



```cmd

uv run --no-project --with pydantic --with scikit-learn --with numpy python -m experiments.franceli.final\_context\_evaluation

```



Run repository tests:



```cmd

uv run --no-project --with pydantic --with pytest pytest tests -q

```



The existing automated test suite passed all 36 tests.



\## Next Steps



1\. Freeze the selected model configuration.

2\. Evaluate once on the held-out test split.

3\. Report final metrics and limitations.

4\. Submit the implementation for team review.



Related GitHub issue: #15.




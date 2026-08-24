# Discussion Intelligence Toolkit

- **Challenge Sponsor:** Independent doctoral research project
- **Challenge Advisor:** Tom Mathews, Doctoral Student, tom.mathews@nyu.edu
- **AI Studio Coach:** Ayush Amberkar, ayush.amberkar@breakthroughtech.org
- **Industry:** Applied Artificial Intelligence and Data Science
- **Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About This Industry

Applied artificial intelligence and data science turn unstructured information into reliable, decision-ready evidence. Practitioners combine data engineering, machine learning, information retrieval, and human review to build systems that support work in domains such as market research, product development, community operations, and research.

This challenge uses discussion intelligence as a concrete market-research application, but its primary purpose is to teach reusable AI engineering skills: modeling structured conversation data, retrieving supporting evidence, evaluating outputs, and communicating limitations.

---

## 🎯 The Challenge

### Project Summary
Build an open-source Python toolkit that converts long asynchronous discussions into structured, evidence-linked insight reports. The target inputs are threaded conversations such as Reddit threads, GitHub discussions, forum posts, or Discord-style exports. The target outputs are reports that identify the main claims in a discussion, group related comments, score discussion quality, and generate grounded summaries that cite specific source comments.

### Why This Matters
Long online discussions contain useful product, research, and community signal, but the signal is expensive to extract manually. This project turns that unstructured discussion data into a workflow that can support synthesis, moderation, community analysis, and retrospective reporting.

### December Deliverables
- An installable Python toolkit, not only notebooks
- A reproducible end-to-end pipeline from raw thread input to final report
- Evaluation results for each major component
- At least one demo notebook or script that runs on student-accessible compute
- A portfolio-ready repository with documentation, results, and example outputs

### Success Criteria
By the end of the semester, the team should be able to demonstrate:
- Ingestion of a threaded discussion into a canonical schema
- Discourse-act or conversation-role labeling on individual utterances
- Quality or usefulness scoring for comments, claims, or discussion segments
- Topic clustering that helps organize large discussions into interpretable themes
- Retrieval of supporting comments for important claims or report sections
- A final insight report in which major claims link back to specific source utterances
- Clear evaluation results, error analysis, and documented limitations

### Technical Scope
The project should be organized into the following components:
1. **Connector / ingestion layer** for threaded discussion data
2. **Canonical schema** for conversations, utterances, and speakers
3. **Discourse classification** for acts such as question, answer, agreement, disagreement, humor, or elaboration
4. **Quality scoring** for arguments, comments, or discussion segments
5. **Embedding + clustering** for topic discovery and report organization
6. **Retrieval** for finding evidence relevant to a report section or generated claim
7. **LLM synthesis** for generating a structured report grounded in retrieved evidence

### Recommended Modeling Progression
Start with simple baselines and add complexity only when evaluation shows a clear benefit:
1. Majority-class or constant-score baseline
2. TF-IDF plus a linear model
3. Sentence-transformer embeddings plus a simple classifier, regressor, or clustering method
4. A compact transformer such as DistilBERT for supervised tasks
5. Retrieval-grounded LLM synthesis after evidence extraction is working

This ordering keeps the project tractable and addresses the evaluation team's request to simplify training work before moving to more complex modeling.

### Project Milestones

Use these milestones to guide your work. Your team should maintain a GitHub Projects board to break these monthly goals into weekly tasks.

| Month | Focus | Expected Outcomes |
|---|---|---|
| **September** | Data understanding and baselines | Dataset selection, reconstructed thread schema, exploratory analysis, baseline models, first evaluation harness |
| **October** | Modeling | Discourse classifier, quality scorer, embedding experiments, clustering experiments, initial error analysis |
| **November** | Integration | End-to-end pipeline, evidence retrieval, report schema, grounded LLM synthesis, package structure |
| **December** | Polish and presentation | Final demo, final results tables, documented limitations, portfolio-ready repository and walkthrough |

> **Note for the team:** Create a GitHub Projects board in this repository and track work by issue. Recommended columns: Backlog, This Week, In Progress, In Review, Done.

---

## 📊 Dataset

### Primary Recommended Data Sources

| Dataset / Source | Purpose in Project | Format | Access |
|---|---|---|---|
| **ConvoKit Coarse Discourse Sequence Corpus** | Initial threaded-discussion path and supervised discourse-act classification | ConvoKit corpus / JSON-like utterance records | https://convokit.cornell.edu/documentation/coarseDiscourse.html |
| **ConvoKit Conversations Gone Awry — ChangeMyView** | Conversation-quality and derailment modeling | ConvoKit corpus / JSON-like utterance records | https://convokit.cornell.edu/documentation/ |
| **Parent-preserving static thread export** | Future end-to-end ingestion, clustering, retrieval, and report generation | JSON / JSONL / CSV after schema validation | The selected export must retain `parent_id`; see `data/README.md` |

### Project Data Access

This repository uses a two-tier data model documented in [`data/README.md`](data/README.md):

1. **Tracked reference data:** [`data/coarse-discourse-reference-sample.jsonl`](data/coarse-discourse-reference-sample.jsonl) contains two complete, source-derived Reddit conversations with reply relationships and discourse-act labels. It is intentionally small enough for schema inspection and smoke tests.
2. **Reproducible complete corpora:** ConvoKit's [Coarse Discourse Sequence Corpus](https://convokit.cornell.edu/documentation/coarseDiscourse.html) supplies the initial threaded-discussion dataset and discourse labels; ConvoKit's Conversations Gone Awry — ChangeMyView corpus supports quality modeling. The full corpora are downloaded on demand and are not committed to this repository.

The prepared Pushshift candidates are not approved for the current connector: their published comments schema does not expose `parent_id`, which is required to reconstruct nested replies. Select and validate a parent-preserving static export before using a bulk Reddit corpus for end-to-end analysis.

### Working Dataset Expectations
- Do **not** try to process the full public Reddit archive. Select a domain-specific subset that fits a student workflow.
- Keep the working dataset for notebooks and experiments small enough to run on Google Colab free-tier hardware. As a rule of thumb, filtered working subsets should stay around or below 1 GB after selection and preprocessing.
- Use the `data/` folder for small metadata files, data dictionaries, sample records, and access notes. Do not commit large raw archives, private data, API keys, or model checkpoints.

### Known Preprocessing and Data Risks
- Reconstruct reply trees so every utterance keeps parent-child context.
- Normalize deleted, removed, empty, or orphaned comments consistently.
- Standardize speaker IDs, thread IDs, timestamps, and utterance IDs.
- Clean markdown, URLs, emojis, and quote formatting without removing argumentative signal.
- Expect domain shift: corpora used for supervised labels may not match the target discussion source used for the demo.

### Data Documentation the Team Must Produce
- Exact source URLs for every dataset actually used
- A short data dictionary for the canonical schema
- Counts for threads, utterances, speakers, and labels retained after filtering
- The train/dev/test split strategy and leakage controls
- A note explaining any licensing or platform usage constraints for the selected data

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP / information extraction / retrieval / grounded summarization

**Recommended Libraries and Tools:**
- Python
- pandas and NumPy
- scikit-learn
- Hugging Face Transformers
- sentence-transformers
- ConvoKit
- FAISS or Chroma for retrieval
- Google Colab for experiments
- `uv` for reproducible local Python environments

### Recommended Report Structure
Each generated insight report should aim to include:
- The main themes or clusters in the discussion
- The strongest claims or arguments within each theme
- Evidence links to the supporting utterances
- Signals about consensus, disagreement, open questions, or unresolved tension
- A short grounded summary with citations to source comments

### How Clustering Should Be Used
Clustering is not the final product by itself. Use it to:
- group related comments into interpretable themes
- help organize long threads before summarization
- surface repeated concerns or recurring claims
- separate consensus areas from disagreement-heavy areas
- make final reports easier to navigate for a reader

### Evaluation Metrics
Use metrics that match each component:

| Component | Primary Metrics | What the Metric Checks |
|---|---|---|
| Discourse-act classifier | Macro F1, per-class F1, confusion matrix | Whether minority labels are handled well |
| Quality scorer | MAE, Spearman correlation | Whether predicted scores match both level and ranking |
| Topic clustering | Silhouette score plus sampled coherence review | Whether clusters are separated and interpretable |
| Retrieval | Recall@k on hand-written thread questions | Whether supporting evidence can be recovered |
| LLM synthesis | Claim coverage, evidence-link accuracy, hallucination rate | Whether reports are useful and grounded |

---

## 📚 Resources to Get Started

The resources below are enough to start productively without overloading the first month.

**Background Reading**
- ConvoKit documentation and corpus model: https://convokit.cornell.edu/documentation/
- Sentence-BERT paper: https://arxiv.org/abs/1908.10084

**Technical Tutorials**
- Hugging Face text classification tutorial: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Sentence Transformers documentation: https://www.sbert.net/
- scikit-learn clustering guide: https://scikit-learn.org/stable/modules/clustering.html

**Code Examples**
- ConvoKit GitHub repository: https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit
- Hugging Face text classification notebook example: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb

**Project Management**
- GitHub Projects documentation: https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects

**Other Useful References**
- sentence-transformers GitHub repository: https://github.com/huggingface/sentence-transformers
- Hugging Face task overview for text classification: https://huggingface.co/tasks/text-classification

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Discord (Break Through Tech workspace) or Email: tom.mathews@nyu.edu  
**Response time:** Within 48 hours on weekdays

**Recommended Tools**
- **Coding:** Google Colab, VS Code, Jupyter notebooks
- **Collaboration:** GitHub Issues, GitHub Projects, Notion
- **Virtual Meetings:** Zoom, Google Meet

### What I Expect From the Team
- Keep work visible in GitHub Issues and the GitHub Projects board
- Record modeling decisions and failed experiments, not only successful ones
- Move reusable code out of notebooks and into Python modules as the project matures
- Keep the repository reproducible and organized enough that an external reviewer can run the demo

---

## 🚀 Getting Started

1. Read this overview and list your open questions before our first meeting.
2. Review `README.md` for the evolving technical plan and repository expectations.
3. Select an initial discussion domain and identify the exact dataset subset you want to use first.
4. Define the canonical conversation schema you will use across connectors, models, and reports.
5. Create your GitHub Projects board and open initial issues for data access, EDA, baselines, and evaluation.

---

## ❓ Questions?

Bring questions to our first meeting during the week of August 24th. Before then, use Slack or email if you hit a blocker on data access, scope, or technical setup.

---

# ELI5 why comment karma is considered inferior to link karma

Source: [link](<https://www.reddit.com/r/explainlikeimfive/comments/169c79/eli5_why_comment_karma_is_considered_inferior_to/>)

Conversation: <code>reddit:t3_169c79</code>

Query: comment karma easier

## At a glance

| | |
|---|---|
| Comments analyzed | 4 |
| Comments quoted | 2 |
| Participants | not recorded in the source |
| Dates | not recorded in the source |
| Themes | not available |
| Discourse labels | not available |
| Civility estimate | not available |

## Key points

> Go to any front page post and then compare the amount of link karma it generated to the amount of comment karma. Almost invariably the latter will be higher. Since more of it is being given out, it follows that it is easier to get and thus inferior.
>
> Source: [reddit:t1\_c7twyqu](#evidence-reddit%3At1_c7twyqu)

> Well normally link karma is usually more hard earned than comment karma, since you have to visit an external website. THIS IS BASED ON SPECULATION
>
> Source: [reddit:t1\_c7twpl2](#evidence-reddit%3At1_c7twpl2)

## Component availability

| Component | Status | Detail |
|---|---|---|
| discourse | unavailable | No fitted classifier supplied. |
| quality | unavailable | No fitted conversation civility scorer supplied. |
| clustering | unavailable | No topic clusterer supplied; excerpts are ungrouped. |
| embeddings | unavailable | No embedder supplied; TF-IDF is used. |
| synthesis | available | Deterministic extractive fallback; no LLM required. |
| retrieval | available | tfidf cosine; 2 selected excerpts. |

## Limitations

- Excerpts represent participants' statements, not verified facts or a complete digest.
- TF-IDF measures lexical relevance; representative excerpts use centroid similarity.
- Labels identify utterance roles; they do not establish consensus, a disagreement target, or whether a question remains open.
- Civility probability applies to the observed conversation context, not per-comment argument strength; domain shift and calibration require evaluation.
- Extractive points contain at most the first 400 characters of a selected utterance; full source text remains in evidence.
- Source timestamps are missing: canonical epoch sentinels are not real dates; evidence dates are null.
- Model inputs retain parent-before-child source order when chronological order cannot be established.
- Source speaker identities are missing; placeholder speaker references do not identify distinct participants.

## Source evidence

Showing 3 of 4 analyzed comments: each quoted comment and the comment it replies to. The JSON report contains all of them.

<a id="evidence-reddit:t3_169c79"></a>

### reddit:t3\_169c79

thread root · speaker not recorded · [conversation](<https://www.reddit.com/r/explainlikeimfive/comments/169c79/eli5_why_comment_karma_is_considered_inferior_to/>)

> I would have thought that people upvoting what you SAID is better than people upvoting what you merely linked.

<a id="evidence-reddit:t1_c7twpl2"></a>

### reddit:t1\_c7twpl2

reply to [reddit:t3\_169c79](#evidence-reddit%3At3_169c79) · speaker not recorded · [conversation](<https://www.reddit.com/r/explainlikeimfive/comments/169c79/eli5_why_comment_karma_is_considered_inferior_to/>)

> Well normally link karma is usually more hard earned than comment karma, since you have to visit an external website. THIS IS BASED ON SPECULATION

<a id="evidence-reddit:t1_c7twyqu"></a>

### reddit:t1\_c7twyqu

reply to [reddit:t3\_169c79](#evidence-reddit%3At3_169c79) · speaker not recorded · [conversation](<https://www.reddit.com/r/explainlikeimfive/comments/169c79/eli5_why_comment_karma_is_considered_inferior_to/>)

> Go to any front page post and then compare the amount of link karma it generated to the amount of comment karma. Almost invariably the latter will be higher. Since more of it is being given out, it follows that it is easier to get and thus inferior.

\# Reddit Coarse Discourse Dataset Finalization



\## Overview



This part of the Market Research 1A Discussion Intelligence Toolkit focuses on cleaning and exploring the Reddit Coarse Discourse dataset before machine learning.



The original dataset contains 115,827 utterances across 9,483 conversations. After cleaning, 100,277 utterances were retained.



\## Dataset Cleaning



The cleaning process is implemented in `cleaning/clean\_data.py` using Python, ConvoKit, and CSV.



The script removes utterances with:

\- Missing majority discourse labels

\- Empty or whitespace-only text

\- Text marked `\[deleted]`

\- Text marked `\[removed]`



The cleaned dataset preserves the utterance ID, conversation ID, speaker ID, reply relationship, comment text, majority discourse label, post depth, and upvote information.



The output is saved locally as `cleaning/cleaned\_discourse\_dataset.csv`.



\## Exploratory Data Analysis



\### Pre-cleaning EDA



The `pre\_cleaning\_eda/eda\_visuals.py` script counts the discourse labels before cleaning and generates a class distribution bar chart.



The `pre\_cleaning\_eda/inspect\_duplicates.py` script is included for duplicate inspection.



\### Post-cleaning EDA



The `cleaning/post\_cleaning\_eda.py` script reads the cleaned CSV and counts the number of utterances in each discourse class.



It generates a second class distribution chart to compare the dataset after cleaning.



\## Final Class Distribution



| Discourse Class | Utterances |

|---|---:|

| Answer | 40,540 |

| Elaboration | 18,880 |

| Question | 16,353 |

| Appreciation | 8,557 |

| Agreement | 4,970 |

| Disagreement | 3,370 |

| Humor | 2,348 |

| Other | 1,972 |

| Negative Reaction | 1,872 |

| Announcement | 1,415 |

| \*\*Total\*\* | \*\*100,277\*\* |



The `answer` category has the largest number of examples, showing that the dataset is imbalanced. This is important when evaluating classification models because high overall accuracy may not reflect performance on smaller classes.



\## Outputs



\- `cleaning/cleaned\_discourse\_dataset.csv`: Cleaned dataset (generated locally)

\- `pre\_cleaning\_eda/discourse\_class\_distribution\_pre\_cleaning.png`: Distribution before cleaning

\- `cleaning/discourse\_class\_distribution\_post\_cleaning.png`: Distribution after cleaning



\## Notes



The cleaned CSV is excluded from Git tracking because it is generated data. The scripts and analysis outputs are maintained in the repository.



The corpus path in the scripts currently points to a local Windows ConvoKit installation and must be adjusted for other environments.



This work supports downstream dataset preparation and discourse classification.


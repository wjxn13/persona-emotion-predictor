# Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events

![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)
![Annotations](https://img.shields.io/badge/Annotations-111k-success.svg)
![Task](https://img.shields.io/badge/Task-Emotion%20Classification-orange.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09162-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.09162)
![ACL 2026 Coming Soon](https://img.shields.io/badge/ACL%202026-Coming%20soon-lightgrey.svg)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/crisyang777/peronsa-e-personality-shaped-emotion-dataset)
![Huggingface Coming Soon](https://img.shields.io/badge/Huggingface-Coming%20soon-lightgrey.svg)


**Feel Free to Contact Us**:  ftyuqin_yang@mail.scut.edu.cn 



## 1. Dataset Summary

> *Two individuals can construe their situations quite similarly (agree on all the facts), and yet react with very different emotions, because they have appraised the adaptational significance of those facts differently.* — Lazarus, Emotion And Adaptation (1991)

Most emotion recognition datasets treat emotions as fixed properties of text—as if a sad story makes everyone feel sad the same way. But human psychology tells a different story: the same event can trigger joy in one person and anxiety in another, depending on their personality.

**Persona-E²** (Persona-Event2Emotion) marks a crucial shift from *"what does the text express?"* to *"how does the text make people feel?"* It is a large-scale, human-grounded dataset capturing how individuals with measured personality traits emotionally react to diverse text-based events. 

The dataset provides a rigorous benchmark for evaluating whether Large Language Models (LLMs) can truly predict and comprehend trait-driven emotional diversity.

<p align="center">
  <img src="Sun_v4.png" width="800">
</p>

## 2. Key Features

* **High Annotation Density:** Every event in the dataset was independently blind-labeled by **36 annotators**. With a total of **111,996 high-quality annotations**, this density allows for a granular analysis of trait-shaped responses.
* **Comprehensive Personality Profiles:** Annotators were profiled using both the **Myers-Briggs Type Indicator (MBTI)** and the **Big Five Inventory (BFI)**. 
* **Cross-Domain Diversity:** The dataset spans three primary domains containing 3,111 filtered events: **News** (factual reports), **Social Media** (informal interactions), and **Life Experience** (quotidian narratives).
* **Advanced Metrics & SDS:** We introduce the **Group Consensus Score (S_consensus)** based on normalized entropy to validate labels. The dataset also features a **Subjective Divergence Subset (SDS)** consisting of 413 highly controversial events, highlighting scenarios where emotional responses are clear but heavily conditioned on personality.
<p align="center">
  <img src="FlowChart_v4.png" width="800">
</p>
## 3. Dataset Structure

The repository consists of 3 core tabular files (CSV format) linking textual events with personality-driven emotion annotations.

### 📁 1. `1_dataset_all_annotators.csv`
Contains the full set of 3,111 events and the raw, fine-grained emotional annotations from all 36 participants.

| Column Name | Description |
| :--- | :--- |
| **`id` / `index`** | Unique identifiers for the textual event. |
| **`chinese_item` / `english_item`**| The bilingual text of the event. |
| **`data_source`** | The original platform. |
| **`category`** | The fine-grained semantic subcategory. |
| **`top1_emotion`** | The General Writer (GW) majority emotion. |
| **`emotion_distribution`** | A JSON string of the writer emotional probability distribution. |
| **`E[1-36]_emo`** | The specific emotion annotated by participant E1 to E36. |
| **`E[1-36]_confidence`**| The self-reported emotional confidence score (1-5). |

### 📁 2. `2_dataset_group_consensus.csv`
Aggregates the individual annotations into 6 distinct BFI-based personality clusters to track in-group alignment.

| Column Name | Description |
| :--- | :--- |
| **`g[1-6]_emo`** | The dominant emotion label within the specific personality group (g1 to g6). |
| **`g[1-6]_members`**| The specific annotator IDs in this cluster. |
| **`g[1-6]_consensus_score`** | The Group Consensus Score. |

### 📁 3. `3_annotator_profiles.csv`
Contains the anonymized demographic information and comprehensive personality profiles for all 36 annotators.

| Column Name | Description |
| :--- | :--- |
| **`ID`** | Anonymized identifier (E1 to E36). |
| **`GNDR`** | Gender of the annotator (M/F). |
| **`MBTI`** | The 16-type Myers-Briggs classification. |
| **`Open. / Cons. / Extra. / Agree. / Neuro.`** | Big Five Inventory (BFI) scores normalized to a continuous scale of [0.0, 1.0]. |

## 4. Event Sources & Statistics

To construct a diverse and comprehensive dataset, we aggregated data from distinct sources spanning formal news, social media discussions, and specific life experience narratives.

| Domain | Source | Link | Count |
| :--- | :--- | :--- | :--- |
| **News** | ABC_news | [Link](https://abcnews.go.com/abcnews) | 130 |
| **News** | BBC_news | [Link](https://www.bbc.com/) | 248 |
| **News** | Independent | [Link](https://www.independent.co.uk/news) | 36 |
| **News** | Today | [Link](https://tophub.today/c/tech) | 314 |
| **News** | The Paper | [Link](https://tophub.today/c/tech) | 273 |
| **News** | Weibo | [Link](https://weibo.com/u/2656274875) | 478 |
| **News** | WeChat | [Link](https://tophub.today/c/tech) | 37 |
| **Social Media** | SocialChem | [Link](https://huggingface.co/datasets/tasksource/social-chemestry-101) | 416 |
| **Social Media** | Reddit | [Link](https://www.reddit.com/) | 440 |
| **Life Experience** | B.E. | [Link](https://www.reddit.com/r/BenignExistence/) | 140 |
| **Life Experience** | FMylife | [Link](https://www.fmylife.com/) | 507 |
| **Life Experience** | IUTB | [Link](http://iusedtobelieve.com/) | 26 |
| **Life Experience** | KindLife | [Link](https://www.randomactsofkindness.org/kindness-stories) | 66 |

## 5. Collection & Annotation Methodology

To transform raw texts into a high-quality psychological benchmark, we implemented a rigorously controlled pipeline:
1.  **3-Stage Event Filtering:** The initial pool of 76,000+ events underwent strict curation: **Content Safety Filtering** (NSFW models) -- **Multi-Dimensional LLM Scoring** (evaluating psychological "differential potential") -- **Expert Verification**.
2.  **Controlled Human Annotation:** 36 screened participants independently labeled all finalized events via a custom online platform.
3.  **Ethics & Privacy:** The entire experimental protocol, including PII anonymization and fair labor compensation, was reviewed and approved by the Institutional Review Board (IRB).

## 6. Quick Start (Python)

You can easily load and explore the dataset using `pandas` and `seaborn`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the core dataset
df_all = pd.read_csv("1_dataset_all_annotators.csv")

# Plot the distribution of the majority vote emotions
plt.figure(figsize=(10, 6))
sns.countplot(data=df_all, x='top1_emotion', 
              order=df_all['top1_emotion'].value_counts().index, 
              palette='viridis')
plt.title('Distribution of Top-1 Emotions (General Reader)')
plt.show()
```

## 7. License & Usage Restrictions

This dataset is released under the **[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license.

* **Academic Use Only:** Restricted to pure academic research, educational purposes, and non-commercial model alignment experiments.
* **ShareAlike:** If you remix, transform, or build upon the material (including fine-tuning LLMs), you must distribute your contributions under the same license.
* **No Commercial Profiling:** Prohibited from being used to train commercial products aimed at monitoring, profiling, or manipulating real users' psychological traits.

## 8. Citation

If you use **Persona-E²** in your research, please cite our ACL paper:

```bibtex
@inproceedings{yang2026personae2,
  title={Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events},
  author={Yang, Yuqin and Zhou, Haowu and Tu, Haoran and Hui, Zhiwen and Yan, Shiqi and Li, HaoYang and She, Dong and Yao, Xianrong and Gao, Yang and Jin, Zhanpeng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  url={[https://arxiv.org/abs/2604.09162](https://arxiv.org/abs/2604.09162)},
  doi={10.48550/arXiv.2604.09162},
  year={2026}
}
```
# Supreme Court Engagement Dataset

This repository contains data and code for analyzing how U.S. Supreme Court dissents “talk with” or “talk past” majority opinions.  
The project combines human-coded engagement scores with outputs from multiple large language models (GPT-5, Claude, DeepSeek, etc.).

---

## 📁 Data Files

The `data/`, `results_filtered/` and `samples/` folders contain the CSV files.  
Large CSVs are stored as ZIP files due to GitHub’s 100MB file limit.

### **Main data files:**

| File | Description |
|------|-------------|
| **data/pair_metadata_w_scores.zip** | Metadata for each majority–dissent pair, including human coder information, case details, and coder scores. |
| **data/30_pairs_w_all_scores.csv** | Full scoring dataset containing human ratings + LLM-generated engagement scores (GPT-5, DeepSeek, Anthropic). |

Unzip these files locally to access the full CSVs.

---

# 🧩 Column Descriptions

## **1. `pair_metadata_w_scores.csv`** (zipped)

| Column | Description |
|--------|-------------|
| **case_key** | Internal ID for linking across datasets. |
| **case_name** | Full case name. |
| **case_name_abbreviation** | Bluebook-style abbreviated case name. |
| **decision_date** | Date the Supreme Court issued the decision. |
| **opinion_type** | `"majority"` or `"dissent"`. |
| **dissent_ind** | Index of the dissenting opinion within the same case. Cases may have multiple dissents; these are indexed 0, 1, 2, … for that case. |
| **majority_text** | Full text of the majority opinion (from CAP). |
| **majority_word_count** | Token count of the majority opinion. |
| **dissent_text** | Full text of the dissenting opinion. |
| **dissent_word_count** | Token count of the dissent opinion. |
| **official citation** | Reporter citation from SCDB/CAP (e.g., U.S. Reports). |
| **precedentAlteration** | Whether the decision alters precedent (SCDB). |
| **issue** | Specific case issue code (SCDB). |
| **issueArea** | Broad issue area (SCDB). Codebook linked below. |
| **decisionDirection** | Ideological direction of the majority (SCDB). |
| **decisionDirectionDissent** | Ideological direction of the dissent (SCDB). |
| **authorityDecision1** | Institutional authority variable (SCDB). |
| **authorityDecision2** | Secondary authority variable (SCDB). |
| **majVotes** | Number of votes on the majority side. |
| **minVotes** | Number of votes on the dissent side. |
| **kldiv_score** | KL divergence between topic distributions of majority vs dissent (higher = more different). |
| **cossim_score** | Cosine similarity between embedding vectors of majority vs dissent (higher = more similar). |

### 🔗 SCDB Documentation
- Full SCDB documentation: http://scdb.wustl.edu/documentation.php  
- e.g. Issue Area codebook: http://scdb.wustl.edu/documentation.php?var=issueArea

---

## **2. `30_pairs_w_all_scores.csv`**

This file extends the above metadata with **human scores** and **LLM scores**.

### **A. Metadata Columns**
| Column | Description |
|--------|-------------|
| **key** | Internal unique index. |
| **row** | Original extraction row index. |
| **case_key** | Link to other datasets. |
| **case_name**, **case_name_abbreviation**, **decision_date** | Case identifiers. |
| **opinion_type**, **dissent_ind** | Opinion metadata (see above). |
| **majority_word_count**, **dissent_word_count** | Word counts. |
| **official citation**, **officialcitation** | Reporter citation fields. |
| **precedentAlteration** | SCDB metadata. |
| **issue**, **issueArea** | Issue-level metadata (SCDB). |
| **decisionDirection**, **decisionDirectionDissent** | Ideological direction variables. |
| **authorityDecision1** | SCDB authority variable. |
| **majVotes**, **minVotes** | Vote counts. |
| **kldiv_score**, **cossim_score** | Topic/embedding similarity metrics. |

---

### **B. Human Engagement Scores**

| Column | Description |
|--------|-------------|
| **cb_score**, **eg_score**, **jm_score**, **st_score**, **sz_score**, **rs_score** | Six coder engagement ratings (1–5). |
| **score_mean** | Mean of the six human coder scores. |
| **binaryRA** | 1 = “talking with”; 0 = “talking past” (thresholded from mean score). |

---

### **C. LLM Engagement Scores**

Each model evaluated all 30 pairs **five times** (responses 0–4). For each model:

#### **OpenAI GPT-5**
- `openai_score_0` … `openai_score_4`  
- `openai_score_mean`

#### **DeepSeek Chat**
- `deepseek_chat_score_0` … `deepseek_chat_score_4`  
- `deepseek_chat_score_mean`

#### **DeepSeek Reasoner**
- `deepseek_reasoner_score_0` … `deepseek_reasoner_score_4`  
- `deepseek_reasoner_score_mean`

#### **Claude Sonnet**
- `anthropic_sonnet_score_0` … `anthropic_sonnet_score_4`  
- `anthropic_sonnet_score_mean`

#### **Claude Opus**
- `anthropic_opus_score_0` … `anthropic_opus_score_4`  
- `anthropic_opus_score_mean`

LLMs use the same 1–5 rubric as human coders.

---

## 📝 Usage

1. **Unzip the CSV files** inside `results_filtered/`.
2. Use the Python scripts in the repo to reproduce:
   - topic models  
   - LLM scoring experiments  
   - correlation matrices  
   - engagement classification  
3. The `samples/` directory contains example model outputs.

---

## 📄 Source of Opinions

Opinions come from:
- **Harvard Caselaw Access Project (CAP)** for full opinion text  
- **Supreme Court Database** for metadata and issue coding  

---

## 📬 Questions?

Feel free to open an issue or contact the maintainer.



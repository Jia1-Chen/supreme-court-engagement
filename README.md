# Supreme Court Engagement Dataset

This repository contains data and code for analyzing how U.S. Supreme Court dissents “talk with” or “talk past” majority opinions.  
The project combines human-coded engagement scores with outputs from multiple large language models (GPT-5, Claude, DeepSeek, etc.).

---

## 📁 Data Files

The `data/` and `results_filtered/` folders contain several zipped CSV files.  
Large CSVs are stored as ZIP files due to GitHub’s 100MB file limit.

### **Main data files:**

| File | Description |
|------|-------------|
| **data/pair_metadata_w_scores.zip** | Metadata for each majority–dissent pair, including human coder information, case details, and coder scores. |
| **data/30_pairs_w_all_scores.zip** | Full scoring dataset containing human ratings + LLM-generated engagement scores (GPT-5, DeepSeek, Anthropic). |

Unzip these files locally to access the full CSVs.

---

## 🧩 Column Descriptions

### ### **1. `pair_metadata_w_scores.csv`**

Case-level metadata and human coder scores.

| Column | Meaning |
|--------|---------|
| `case_key` | Internal ID linking cases across datasets. |
| `case_name` | Full case name from the Supreme Court opinion. |
| `case_name_abbreviation` | Shortened Bluebook-style case name. |
| `decision_date` | Date the Supreme Court issued the decision. |
| `opinion_type` | `"majority"` or `"dissent"` (this dataset contains dissents). |
| `dissent_ind` | 1 if the row is a dissent. |
| `justice` | Justice authoring the opinion. |
| `vote_majority` | Number of justices in the majority. |
| `vote_minority` | Number of dissenting justices. |
| `maj_court_dir`, `min_court_dir` | Direction codes (Supreme Court Database). |
| `issue`, `issue_area` | Issue categorizations from SCDB. |
| `precedent_alteration` | Whether the case alters precedent. |
| `party_winning` | Whether the petitioning party won. |
| `majVotes`, `minVotes` | Vote counts for majority/dissent. |
| `naturalCourt` | SCDB natural court identifier. |
| **Human coder scores:** |
| `cb_score`, `eg_score`, `jm_score`, `st_score`, `sz_score`, `rs_score` | Engagement scores (1–5) from six human coders. |
| `cb_reasoning`, `eg_reasoning`, `jm_reasoning`, … | Rationale written by each coder. |
| `score_mean` | Average of the six human scores. |
| **Binary variable:** |
| `binaryRA` | 1 = “talking with”; 0 = “talking past” (thresholded from human mean score). |

---

### ### **2. `30_pairs_w_all_scores.csv`**

This dataset includes **all human scores + all LLM scores**, repeated across multiple runs.

#### **Columns include:**

### **Basic case info**
| Column | Meaning |
|--------|---------|
| `case_key`, `case_name`, `decision_date` | Basic metadata linking to SCDB and opinion text. |

### **Human scores**
Same as above:  
`cb_score`, `eg_score`, `jm_score`, `st_score`, `sz_score`, `rs_score`, and the associated reasonings.

### **LLM Scores (5 repeated runs each)**
Each model provides 5 scores, plus a column for the mean:

#### **OpenAI GPT-5**
- `openai_score_0` … `openai_score_4`
- `openai_score_mean`

#### **DeepSeek Chat**
- `deepseek_chat_score_0` … `deepseek_chat_score_4`
- `deepseek_chat_score_mean`

#### **DeepSeek Reasoner**
- `deepseek_reasoner_score_0` … `deepseek_reasoner_score_4`
- `deepseek_reasoner_score_mean`

#### **Anthropic Claude Sonnet**
- `anthropic_sonnet_score_0` … `anthropic_sonnet_score_4`
- `anthropic_sonnet_score_mean`

#### **Anthropic Claude Opus**
- `anthropic_opus_score_0` … `anthropic_opus_score_4`
- `anthropic_opus_score_mean`

Together, these allow fine-grained comparison of:
- human engagement assessments  
- model variability  
- model–human agreement (raw & rank correlations)

---

## 📝 Usage

1. **Unzip the CSV files** inside `data/` and `results_filtered/`.
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
- **Supreme Court Database (Spaeth et al.)** for metadata and issue coding  

---

## 📬 Questions?

Feel free to open an issue or contact the maintainer.



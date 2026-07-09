# Supreme Court Engagement Dataset

This repository contains data and code for analyzing how U.S. Supreme Court dissents “talk with” or “talk past” majority opinions.

The project combines computational text-similarity measures, human-coded engagement scores, and outputs from multiple large language models, including GPT-5, Claude, and DeepSeek.

---

## Repository Structure

| Path | Description |
|------|-------------|
| `data/` | Raw data files used in the replication pipeline. |
| `results_filtered/` | Main processed outputs for the filtered majority–dissent corpus, including topic-model, doc2vec, Sentence-BERT, Legal-BERT, and merged metadata files. |
| `samples/` | Smaller sample files and LLM scoring outputs used for validation. |
| `topic_model_filtered.py` | Reproduces the LDA topic-model measure. |
| `doc2vec_filtered.py` | Reproduces the doc2vec embedding measure. |
| `SBERT_filtered.py` | Reproduces the Sentence-BERT embedding measure. |
| `LegalBERT_filtered.py` | Reproduces the Legal-BERT embedding measure. |
| `openai_sample.py` | Runs GPT-5 scoring on the validation sample. |
| `anthropic_sample_opus.py` | Runs Claude Opus scoring on the validation sample. |
| `anthropic_sample_sonnet.py` | Runs Claude Sonnet scoring on the validation sample. |
| `deepseek_sample_chat.py` | Runs DeepSeek Chat scoring on the validation sample. |
| `deepseek_sample_reasoner.py` | Runs DeepSeek Reasoner scoring on the validation sample. |
| `data_download_replication.py` | Demonstrates the data download and opinion-pair construction pipeline. |

Large CSV files are stored as ZIP files when needed because GitHub does not allow files larger than 100 MB.

---

## Main Data Files

| File | Description |
|------|-------------|
| `results_filtered/pair_metadata_w_scores.zip` | Zipped pair-level dataset containing majority–dissent metadata, opinion text, SCDB variables, human scores where available, and computed engagement measures. |
| `samples/30_pairs_with_all_scores.csv` | Validation sample containing 30 majority–dissent pairs with human ratings and repeated LLM-generated engagement scores. |

Unzip `results_filtered/pair_metadata_w_scores.zip` locally to access the full CSV.

```bash
unzip results_filtered/pair_metadata_w_scores.zip -d results_filtered/
```

---

## Column Descriptions

## 1. `pair_metadata_w_scores.csv`

This file is stored as:

```text
results_filtered/pair_metadata_w_scores.zip
```

It contains the full filtered majority–dissent pair dataset.

| Column | Description |
|--------|-------------|
| `case_key` | Internal ID for linking across datasets. |
| `case_name` | Full case name. |
| `case_name_abbreviation` | Abbreviated case name. |
| `decision_date` | Date the Supreme Court issued the decision. |
| `opinion_type` | Opinion type field. |
| `dissent_ind` | Index of the dissenting opinion within the same case. Cases may have multiple dissents; these are indexed 0, 1, 2, etc. |
| `majority_text` | Full text of the majority opinion from the Caselaw Access Project. |
| `majority_word_count` | Word count of the majority opinion. |
| `dissent_text` | Full text of the dissenting opinion. |
| `dissent_word_count` | Word count of the dissenting opinion. |
| `official citation` / `officialcitation` | Reporter citation fields used for linking CAP and SCDB records. |
| `precedentAlteration` | Whether the decision alters precedent, from SCDB. |
| `issue` | Specific case issue code from SCDB. |
| `issueArea` | Broad issue-area code from SCDB. |
| `decisionDirection` | Ideological direction of the majority decision, from SCDB. |
| `decisionDirectionDissent` | Ideological direction of the dissent, from SCDB. |
| `authorityDecision1` | Institutional authority variable from SCDB. |
| `authorityDecision2` | Secondary authority variable from SCDB. |
| `majVotes` | Number of votes on the majority side. |
| `minVotes` | Number of votes on the dissenting side. |
| `kldiv_score` | KL divergence between LDA topic distributions of majority and dissent. Higher values indicate greater topical divergence. |
| `d2v_cossim_score` | Cosine similarity between doc2vec embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `sbert_cossim_score` | Cosine similarity between Sentence-BERT embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `legalbert_cossim_score` | Cosine similarity between Legal-BERT embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `d2v_cosdist_score` | Cosine distance between doc2vec embeddings of majority and dissent, computed as `1 - d2v_cossim_score`. Higher values indicate greater textual divergence. |
| `sbert_cosdist_score` | Cosine distance between Sentence-BERT embeddings of majority and dissent, computed as `1 - sbert_cossim_score`. Higher values indicate greater textual divergence. |
| `legalbert_cosdist_score` | Cosine distance between Legal-BERT embeddings of majority and dissent, computed as `1 - legalbert_cossim_score`. Higher values indicate greater textual divergence. |

### SCDB Documentation

- Full SCDB documentation: <http://scdb.wustl.edu/documentation.php>
- Issue Area codebook: <http://scdb.wustl.edu/documentation.php?var=issueArea>

---

## 2. `samples/30_pairs_with_all_scores.csv`

This file contains the 30-pair validation sample with human scores, LLM scores, and computational measures.

### A. Metadata Columns

| Column | Description |
|--------|-------------|
| `key` | Internal unique index. |
| `row` | Original extraction row index. |
| `case_key` | Link to other datasets. |
| `case_name`, `case_name_abbreviation`, `decision_date` | Case identifiers. |
| `opinion_type`, `dissent_ind` | Opinion metadata. |
| `majority_text` | Full text of the majority opinion. |
| `dissent_text` | Full text of the dissenting opinion. |
| `majority_word_count`, `dissent_word_count` | Opinion word counts. |
| `official citation`, `officialcitation` | Reporter citation fields. |
| `precedentAlteration` | SCDB precedent-alteration variable. |
| `issue`, `issueArea` | SCDB issue variables. |
| `decisionDirection`, `decisionDirectionDissent` | SCDB ideological-direction variables. |
| `authorityDecision1`, `authorityDecision2` | SCDB authority variables. |
| `majVotes`, `minVotes` | Vote counts. |

### B. Computational Measure Columns

| Column | Description |
|--------|-------------|
| `kldiv_score` | KL divergence between LDA topic distributions of majority and dissent. Higher values indicate greater topical divergence. |
| `d2v_cossim_score` | Cosine similarity between doc2vec embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `sbert_cossim_score` | Cosine similarity between Sentence-BERT embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `legalbert_cossim_score` | Cosine similarity between Legal-BERT embeddings of majority and dissent. Higher values indicate greater textual similarity. |
| `d2v_cosdist_score` | Cosine distance between doc2vec embeddings of majority and dissent. Higher values indicate greater textual divergence. |
| `sbert_cosdist_score` | Cosine distance between Sentence-BERT embeddings of majority and dissent. Higher values indicate greater textual divergence. |
| `legalbert_cosdist_score` | Cosine distance between Legal-BERT embeddings of majority and dissent. Higher values indicate greater textual divergence. |

### C. Human Engagement Scores

| Column | Description |
|--------|-------------|
| `cb_score`, `eg_score`, `jm_score`, `st_score`, `sz_score`, `rs_score` | Human coder engagement ratings on a 1–5 scale. |
| `score_mean` | Mean human coder score. |
| `binaryRA` | Binary engagement indicator derived from human scores. |

### D. LLM Engagement Scores

Each LLM evaluates each of the 30 validation pairs five times. Columns follow the pattern `model_score_0` through `model_score_4`, plus a model-specific mean score.

| Model | Columns |
|-------|---------|
| GPT-5 | `openai_score_0` … `openai_score_4`, `openai_score_mean` |
| DeepSeek Chat | `deepseek_chat_score_0` … `deepseek_chat_score_4`, `deepseek_chat_score_mean` |
| DeepSeek Reasoner | `deepseek_reasoner_score_0` … `deepseek_reasoner_score_4`, `deepseek_reasoner_score_mean` |
| Claude Sonnet | `anthropic_sonnet_score_0` … `anthropic_sonnet_score_4`, `anthropic_sonnet_score_mean` |
| Claude Opus | `anthropic_opus_score_0` … `anthropic_opus_score_4`, `anthropic_opus_score_mean` |

All human and LLM scores use the same 1–5 rubric, where higher scores indicate greater deliberative engagement.

---

## Computational Measures

The repository includes four main computational measures of the textual relationship between each majority opinion and dissenting opinion.

### 1. Topic Model

Script:

```bash
python topic_model_filtered.py
```

This script fits an LDA topic model to the filtered opinion corpus and represents each opinion as a topic distribution. Majority–dissent divergence is measured using KL divergence.

Outputs are stored in:

```text
results_filtered/topic_model/
```

The main pair-level score is:

```text
kldiv_score
```

Higher values indicate greater topical divergence between the majority and dissent.

---

### 2. Doc2vec

Script:

```bash
python doc2vec_filtered.py
```

This script trains a doc2vec model over majority and dissenting opinions and computes majority–dissent similarity in the learned document-embedding space.

Outputs are stored in:

```text
results_filtered/doc2vec/
```

The main pair-level scores are:

```text
d2v_cossim_score
d2v_cosdist_score
```

`d2v_cossim_score` is the cosine similarity between the majority and dissent document embeddings.

`d2v_cosdist_score` is the corresponding cosine distance.

---

### 3. Sentence-BERT

Script:

```bash
python SBERT_filtered.py
```

The Sentence-BERT script uses:

```text
sentence-transformers/all-distilroberta-v1
```

Long opinions are split into token chunks using the GPT-2 tokenizer from `tiktoken`. Each chunk has at most 512 tokens with a 50-token overlap. The script embeds each chunk using Sentence-BERT with normalized embeddings, then averages chunk embeddings to obtain one document-level embedding per opinion.

Outputs are stored in:

```text
results_filtered/sbert/
```

Main outputs include:

| File | Description |
|------|-------------|
| `results_filtered/sbert/document_mapping.csv` | Mapping from document index to case and opinion label. |
| `results_filtered/sbert/document_embeddings.npy` | Document-level Sentence-BERT embeddings. |
| `results_filtered/sbert/cosine_similarities.npy` | Majority–dissent cosine similarities. |
| `results_filtered/sbert/cosine_similarity_metadata.csv` | Pair-level cosine similarity metadata. |

The main pair-level scores are:

```text
sbert_cossim_score
sbert_cosdist_score
```

`sbert_cossim_score` is the cosine similarity between the majority and dissent Sentence-BERT embeddings.

`sbert_cosdist_score` is the corresponding cosine distance.

---

### 4. Legal-BERT

Script:

```bash
python LegalBERT_filtered.py
```

The Legal-BERT script uses:

```text
nlpaueb/legal-bert-base-uncased
```

Long opinions are split using the Legal-BERT tokenizer into chunks of at most 512 tokens, including special tokens, with a 50-token overlap. For each chunk, the script extracts the final hidden states and mean-pools token embeddings using the attention mask. Chunk embeddings are L2-normalized, averaged into a document-level embedding, and the final document embedding is normalized again.

Outputs are stored in:

```text
results_filtered/legalbert/
```

Main outputs include:

| File | Description |
|------|-------------|
| `results_filtered/legalbert/document_mapping.csv` | Mapping from document index to case and opinion label. |
| `results_filtered/legalbert/document_embeddings.npy` | Document-level Legal-BERT embeddings. |
| `results_filtered/legalbert/cosine_similarities.npy` | Majority–dissent cosine similarities. |
| `results_filtered/legalbert/cosine_similarity_metadata.csv` | Pair-level cosine similarity metadata. |

The main pair-level scores are:

```text
legalbert_cossim_score
legalbert_cosdist_score
```

`legalbert_cossim_score` is the cosine similarity between the majority and dissent Legal-BERT embeddings.

`legalbert_cosdist_score` is the corresponding cosine distance.

---

## LLM Scoring

The repository includes scripts for scoring the 30-pair validation sample with multiple LLMs.

| Script | Model family |
|--------|-------------|
| `openai_sample.py` | OpenAI GPT-5 |
| `anthropic_sample_opus.py` | Claude Opus |
| `anthropic_sample_sonnet.py` | Claude Sonnet |
| `deepseek_sample_chat.py` | DeepSeek Chat |
| `deepseek_sample_reasoner.py` | DeepSeek Reasoner |

Each script asks the model to rate whether the dissent is “talking with” or “talking past” the majority on a 1–5 scale and stores the resulting score and reasoning.

---

## Usage

A typical workflow is:

1. Clone the repository.

```bash
git clone git@github.com:Jia1-Chen/supreme-court-engagement.git
cd supreme-court-engagement
```

2. Unzip the main processed dataset.

```bash
unzip results_filtered/pair_metadata_w_scores.zip -d results_filtered/
```

3. Run the desired replication scripts.

```bash
python topic_model_filtered.py
python doc2vec_filtered.py
python SBERT_filtered.py
python LegalBERT_filtered.py
```

4. Use `samples/30_pairs_with_all_scores.csv` to reproduce validation analyses involving human and LLM scores.

---

## Source of Opinions

Opinion texts come from the Harvard Caselaw Access Project.

Case metadata and issue coding come from the Supreme Court Database.

---

## Notes on Interpretation

The computational measures are oriented in two different ways:

- Similarity scores, such as `d2v_cossim_score`, `sbert_cossim_score`, and `legalbert_cossim_score`, are higher when majority and dissent are more textually similar.
- Distance or divergence scores, such as `kldiv_score`, `d2v_cosdist_score`, `sbert_cosdist_score`, and `legalbert_cosdist_score`, are higher when majority and dissent are more textually divergent.

In the engagement interpretation used in this project, greater textual divergence is generally interpreted as less deliberative engagement, while greater textual similarity is generally interpreted as more deliberative engagement.

---

## Questions

Please open an issue or contact the maintainer.

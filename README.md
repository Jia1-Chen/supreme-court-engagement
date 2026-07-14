# Supreme Court Engagement Dataset

This repository contains data and code for analyzing how U.S. Supreme Court dissents “talk with” or “talk past” majority opinions.

The project combines computational text-similarity measures, human-coded engagement scores, and outputs from multiple large language models, including GPT-5, Claude, and DeepSeek.

---

## Repository Structure

| Path | Description |
|---|---|
| `data/` | Raw-data workspace used by the opinion-download and preprocessing pipeline. |
| `results_filtered/` | Processed data and model outputs for the filtered majority–dissent corpus. |
| `results_filtered/pair_metadata.zip` | Majority–dissent metadata with the original majority and dissent opinion text, but without the final computational scores. Use this file when rerunning the computational-measure scripts. |
| `results_filtered/pair_metadata_w_scores.csv` | Main analysis dataset containing case metadata, computational scores, standardized retained measures, and the final composite divergence measure. |
| `samples/` | Validation sample and LLM-scoring outputs. |
| `data_download_replication.py` | Demonstrates downloading opinion data and constructing majority–dissent pairs. |
| `topic_model_filtered.py` | Reproduces the LDA/KL-divergence measure. |
| `doc2vec_filtered.py` | Reproduces the doc2vec measure. |
| `SBERT_filtered.py` | Reproduces the Sentence-BERT measure evaluated during model selection. |
| `LegalBERT_filtered.py` | Reproduces the Legal-BERT measure evaluated during model selection. |
| `voyage_filtered.py` | Generates VoyageAI embeddings and majority–dissent cosine distances. |
| `openai_sample.py` | Runs GPT-5 scoring on the validation sample. |
| `anthropic_sample_opus.py` | Runs Claude Opus scoring on the validation sample. |
| `anthropic_sample_sonnet.py` | Runs Claude Sonnet scoring on the validation sample. |
| `deepseek_sample_chat.py` | Runs DeepSeek Chat scoring on the validation sample. |
| `deepseek_sample_reasoner.py` | Runs DeepSeek Reasoner scoring on the validation sample. |

---

## Main Data Files

### `results_filtered/pair_metadata_w_scores.csv`

This is the main pair-level dataset used in the revised analysis. It contains one row per majority–dissent pair.

Unlike `pair_metadata.zip`, this file does not include the full opinion text. Instead, it contains the metadata and computational variables needed for the final statistical analysis.

| Column | Description |
|---|---|
| `case_key` | Internal case identifier. |
| `case_name_abbreviation` | Abbreviated case name. |
| `dissent_ind` | Index of the dissent within the case. |
| `majority_word_count` | Word count of the majority opinion. |
| `dissent_word_count` | Word count of the dissenting opinion. |
| `officialcitation` | Official reporter citation. |
| `issuearea` | Supreme Court Database issue-area code. |
| `majvotes` | Number of votes in the majority. |
| `minvotes` | Number of votes in dissent. |
| `year` | Decision year. |
| `kldiv_score` | KL divergence between the majority and dissent topic distributions. |
| `d2v_cosdist_score` | Cosine distance between the majority and dissent doc2vec embeddings. |
| `sbert_cosdist_score` | Cosine distance between the majority and dissent Sentence-BERT embeddings. Evaluated but not retained in the final composite. |
| `legalbert_cosdist_score` | Cosine distance between the majority and dissent Legal-BERT embeddings. Evaluated but not retained in the final composite. |
| `voyage3_cosdist` | Cosine distance between the majority and dissent Voyage 3 Large embeddings. |
| `voyagelaw_cosdist` | Cosine distance between the majority and dissent Voyage Law 2 embeddings. Evaluated but not retained in the final composite. |
| `z_kldiv` | Standardized KL-divergence score. |
| `z_d2v` | Standardized doc2vec cosine-distance score. |
| `z_voyage3` | Standardized Voyage 3 Large cosine-distance score. |
| `composite_div` | Final composite divergence measure constructed from `z_kldiv`, `z_d2v`, and `z_voyage3`. |

Higher distance or divergence values indicate that the majority and dissent are farther apart in the corresponding representation.

### `results_filtered/pair_metadata.zip`

This archive contains the majority–dissent metadata together with the original majority and dissent opinion text, but without the final computational scores.

Use this file when you want to rerun the computational-measure scripts, because those scripts require the original text fields, including:

```text
majority_text
dissent_text
```

Extract it with:

```bash
unzip results_filtered/pair_metadata.zip -d results_filtered/
```

This should produce:

```text
results_filtered/pair_metadata.csv
```

The computational scripts read `pair_metadata.csv` and write their model-specific outputs under `results_filtered/`.

### `samples/30_pairs_with_all_scores.csv`

This is the canonical 30-pair validation sample. It contains case metadata, human engagement ratings, repeated LLM-generated scores, and computational measures.

Each LLM evaluates each pair five times. Model columns generally follow the pattern `model_score_0` through `model_score_4`, together with a model-specific mean.

---

## Computational Measures

The final composite divergence measure retains three components:

1. Topic-model divergence
2. Doc2vec cosine distance
3. Voyage 3 Large cosine distance

Their standardized values are stored as:

```text
z_kldiv
z_d2v
z_voyage3
```

and combined in:

```text
composite_div
```

Sentence-BERT, Legal-BERT, and Voyage Law 2 were also evaluated during model selection but were not retained in the final composite. Their outputs remain in the repository for transparency and comparison.

### Topic model

Run:

```bash
python topic_model_filtered.py
```

The topic-model script represents each opinion as a topic distribution and computes KL divergence between the majority and dissent.

The main score is:

```text
kldiv_score
```

### Doc2vec

Run:

```bash
python doc2vec_filtered.py
```

The main score is:

```text
d2v_cosdist_score
```

### Sentence-BERT

Run:

```bash
python SBERT_filtered.py
```

The main score is:

```text
sbert_cosdist_score
```

Sentence-BERT was evaluated but not retained in the final composite measure.

### Legal-BERT

Run:

```bash
python LegalBERT_filtered.py
```

The main score is:

```text
legalbert_cosdist_score
```

Legal-BERT was evaluated but not retained in the final composite measure.

---

## VoyageAI

Run:

```bash
python voyage_filtered.py
```

The script uses two VoyageAI embedding models:

```text
voyage-3-large
voyage-law-2
```

It embeds each majority and dissent opinion and computes cosine distance for each pair.

The main scores are:

```text
voyage3_cosdist
voyagelaw_cosdist
```

`voyage3_cosdist` is retained in the final composite measure. `voyagelaw_cosdist` was evaluated but not retained.

The script reads:

```text
results_filtered/pair_metadata.csv
```

and writes its outputs to:

```text
results_filtered/voyage/
```

Running the script requires the `voyageai` package and a `VOYAGE_API_KEY`.

---

## LLM Validation Scores

The repository includes scripts for scoring the 30-pair validation sample with several LLM families.

| Script | Model family |
|---|---|
| `openai_sample.py` | OpenAI GPT-5 |
| `anthropic_sample_opus.py` | Claude Opus |
| `anthropic_sample_sonnet.py` | Claude Sonnet |
| `deepseek_sample_chat.py` | DeepSeek Chat |
| `deepseek_sample_reasoner.py` | DeepSeek Reasoner |

All human and LLM engagement ratings use the same 1–5 rubric, where higher values indicate greater deliberative engagement.

Running these scripts requires the relevant provider credentials and may incur API charges.

---

## Reconstructing the Opinion-Pair Data

`data_download_replication.py` demonstrates how to:

1. Download U.S. Supreme Court archives from the Harvard Caselaw Access Project.
2. Extract case and opinion text.
3. Construct majority–dissent pairs.
4. Optionally merge the resulting pairs with Supreme Court Database variables.

Run:

```bash
python data_download_replication.py
```

Large CAP archives are downloaded into:

```text
data/zip/
```

These downloaded archives are generated locally and should not be committed to Git.

The optional Supreme Court Database merge is a template and may require adapting local filenames and citation-field names.

---

## Typical Workflow

Clone the repository:

```bash
git clone git@github.com:Jia1-Chen/supreme-court-engagement.git
cd supreme-court-engagement
```

For analysis using the already-computed measures, use:

```text
results_filtered/pair_metadata_w_scores.csv
```

For rerunning the text-based computational scripts, extract:

```text
results_filtered/pair_metadata.zip
```

and use the resulting:

```text
results_filtered/pair_metadata.csv
```

For the human and LLM validation analyses, use:

```text
samples/30_pairs_with_all_scores.csv
```

---

## Source Data

Opinion texts come from the Harvard Caselaw Access Project.

Case metadata and issue coding come from the Supreme Court Database.

Users should consult the source projects for their documentation, coverage, licensing, and citation requirements.

---

## Notes on Interpretation

The main computational variables are distances or divergences. Higher values indicate a greater representational difference between the majority and dissent.

In this project, greater divergence is generally interpreted as less deliberative engagement, while lower divergence is generally interpreted as greater engagement. This interpretation should be considered together with the human-coded and LLM-based validation analyses rather than treating any single computational measure as a direct behavioral label.

---

## Reproducibility Notes

- The main scored dataset does not include the original opinion text.
- `pair_metadata.zip` provides the opinion text needed to rerun the computational scripts.
- External APIs and hosted model versions may change.
- VoyageAI embedding generation requires `VOYAGE_API_KEY` and may incur charges.
- Saved VoyageAI embedding files are relatively large.
- Raw CAP archives downloaded into `data/zip/` should remain untracked.
- Randomized models and external LLM calls may not produce byte-identical outputs unless model versions, provider behavior, package versions, and random seeds are fixed.

---

## Questions

Please open an issue or contact the maintainer.
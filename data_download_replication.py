# ============================================================
# Replication Notebook for Supreme Court Opinion Data
# ============================================================

import os
import json
import zipfile
import requests
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Download CAP Supreme Court archives
# ------------------------------------------------------------

CAP_INDEX = "https://static.case.law/us/"
os.makedirs("data/zip", exist_ok=True)

index_html = requests.get(CAP_INDEX).text
zip_urls = [
    CAP_INDEX + line.split('"')[1]
    for line in index_html.splitlines()
    if line.strip().endswith(".zip</a>")
]

print("CAP yearly archives found:", len(zip_urls))

for url in tqdm(zip_urls, desc="Downloading CAP ZIP files"):
    fname = url.split("/")[-1]
    path = os.path.join("data", "zip", fname)
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

# ------------------------------------------------------------
# 2. Load all CAP JSON cases
# ------------------------------------------------------------

cases = []

for zfile in tqdm(os.listdir("data/zip"), desc="Extracting JSON cases"):
    zpath = os.path.join("data", "zip", zfile)
    with zipfile.ZipFile(zpath, "r") as z:
        for name in z.namelist():
            if name.endswith(".json"):
                try:
                    cases.append(json.loads(z.read(name)))
                except Exception:
                    pass

print("Total CAP cases loaded:", len(cases))

# ------------------------------------------------------------
# 3. Compute majority length (in words) per case
# ------------------------------------------------------------

majority_lengths = []

for case in tqdm(cases, desc="Computing majority opinion lengths"):
    case_key = case.get("id")
    opinions = case.get("casebody", {}).get("opinions", [])

    # Find majority opinion, if any
    maj_texts = [
        op.get("text", "")
        for op in opinions
        if op.get("type", "").lower() == "majority"
    ]
    if not maj_texts:
        continue

    majority_text = maj_texts[0]
    # Word-level length: split on whitespace
    majority_length = len(majority_text.split())

    majority_lengths.append({
        "case_key": case_key,
        "majority_length": majority_length
    })

majority_length_df = pd.DataFrame(majority_lengths)
os.makedirs("results", exist_ok=True)
majority_length_df.to_csv("results/majority_length.csv", index=False)

print("majority_length.csv saved with shape:", majority_length_df.shape)

# Build dictionary for quick lookup
majority_length_dict = majority_length_df.set_index("case_key")["majority_length"].to_dict()

# ------------------------------------------------------------
# 4. Helper functions for citations
# ------------------------------------------------------------

def _unique_preserve(seq):
    """Deduplicate while preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def collect_cites_by_opinion(case_obj):
    """
    Map opinion_index -> list of citation strings for that opinion.

    Uses 'cites_to' either at the top level or under casebody.data.cites_to,
    skipping unattributed citations (opinion_index == -1).
    """
    cites_list = (
        case_obj.get("cites_to")
        or case_obj.get("casebody", {}).get("data", {}).get("cites_to")
        or []
    )
    by_idx = defaultdict(list)
    for c in cites_list:
        oi = c.get("opinion_index")
        if oi is None or oi == -1:
            continue
        cite_str = c.get("cite") or c.get("citation") or c.get("normalized_cite")
        if cite_str:
            by_idx[oi].append(cite_str)
    return {k: _unique_preserve(v) for k, v in by_idx.items()}

# ------------------------------------------------------------
# 5. Build majority–dissent pairs (pair_metadata.csv)
# ------------------------------------------------------------

pair_rows = []

for case in tqdm(cases, desc="Building majority–dissent pairs"):
    case_key = case.get("id")
    opinions = case.get("casebody", {}).get("opinions", [])

    # We only keep cases where the majority has more than 50 words
    if majority_length_dict.get(case_key, 0) <= 50:
        continue

    # Check if there is at least one dissent
    if not any(op.get("type", "").lower() == "dissent" for op in opinions):
        continue

    # Identify majority opinion index and text
    maj_idx = next(
        i for i, op in enumerate(opinions)
        if op.get("type", "").lower() == "majority"
    )
    majority_text = opinions[maj_idx].get("text", "")

    # Identify all dissent indices
    dissent_indices = [
        i for i, op in enumerate(opinions)
        if op.get("type", "").lower() == "dissent"
    ]

    # Opinion-level citations
    cites_list = (
        case.get("cites_to")
        or case.get("casebody", {}).get("data", {}).get("cites_to")
        or []
    )
    cites_by_index = collect_cites_by_opinion(case)

    # Count unattributed citations (opinion_index == -1)
    unattributed_cites_count = sum(
        1 for c in cites_list if c.get("opinion_index") == -1
    )

    majority_cites = cites_by_index.get(maj_idx, [])

    for j, diss_idx in enumerate(dissent_indices, start=1):
        dissent_text = opinions[diss_idx].get("text", "")
        dissent_cites = cites_by_index.get(diss_idx, [])

        pair_rows.append({
            "case_key": case_key,
            "case_name": case.get("name"),
            "case_name_abbreviation": case.get("name_abbreviation"),
            "decision_date": case.get("decision_date"),
            "opinion_type": "dissent",
            "dissent_ind": j,
            "majority_text": majority_text,
            "dissent_text": dissent_text,
            "majority_cites": majority_cites,
            "dissent_cites": dissent_cites,
            "unattributed_cites_count": unattributed_cites_count,
        })

pair_metadata_df = pd.DataFrame(pair_rows)
os.makedirs("results_filtered", exist_ok=True)
pair_metadata_df.to_csv("results_filtered/pair_metadata.csv", index=False)

print("pair_metadata.csv saved with shape:", pair_metadata_df.shape)

# ------------------------------------------------------------
# 6. Optional: merge with SCDB for users who want full metadata
# ------------------------------------------------------------

# This section is optional. It demonstrates how to merge pair_metadata.csv
# with SCDB's case-centered citation file. It assumes you have extracted
# official or sct cites from CAP into case-level metadata elsewhere.

# Example template (to be customized to your local citation fields):

scdb = pd.read_csv("SCDB_2024_01_caseCentered_Citation.csv", dtype=str)
pair_metadata_df = pd.read_csv("results_filtered/pair_metadata.csv")

merged = pair_metadata_df.merge(
    scdb,
    left_on="official_cite",   # or another column you derive from CAP
    right_on="usCite",
    how="left"
)
merged.to_csv("results_filtered/pair_metadata_with_scdb.csv", index=False)
print("Merged pair_metadata_with_scdb.csv saved with shape:", merged.shape)

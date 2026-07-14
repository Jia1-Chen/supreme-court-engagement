"""
Embed SCOTUS majority and dissent opinions with Voyage (whole-document, no chunking),
compute majority-dissent cosine distance for two models.

- Reads:  results_filtered/pair_metadata.csv
          (case_key, dissent_ind, majority_text, dissent_text)
- Writes: results_filtered/voyage/voyage_cosdist.csv
          (pair_id, case_key, dissent_ind, voyage3_cosdist, voyagelaw_cosdist)

Safe to stop and restart: embeddings are checkpointed to
results_filtered/voyage/emb_*.pkl and resumed automatically.
Run again after an interruption and it picks up where it left off.

To TEST first, set LIMIT = 20 below, run, check the output looks sane, then set
LIMIT = None and run the full corpus.
"""

import os, time, pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import voyageai

# ------------------------------------------------------------------ config
LIMIT = None          # set to e.g. 20 to test on a few pairs first; None = full corpus
DATA = "results_filtered/pair_metadata.csv"
OUTDIR = "results_filtered/voyage"
MODELS = {"voyage3": "voyage-3-large", "voyagelaw": "voyage-law-2"}
TOKEN_BUDGET = 60_000    # generous safety margin under the 120K hard cap
ITEM_CAP = 32            # fewer texts per request so long ones can't stack
# ------------------------------------------------------------------

load_dotenv()
vo = voyageai.Client()   # reads VOYAGE_API_KEY from environment / .env
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(DATA).reset_index(drop=True)
if LIMIT:
    df = df.head(LIMIT).copy()
print(f"loaded {len(df)} pairs")

# one entry per opinion: id is stable across runs -> "{rowindex}_maj" / "{rowindex}_dis"
opinions = []
for i, row in df.iterrows():
    maj = str(row["majority_text"]) if pd.notna(row["majority_text"]) else ""
    dis = str(row["dissent_text"]) if pd.notna(row["dissent_text"]) else ""
    opinions.append((f"{i}_maj", maj))
    opinions.append((f"{i}_dis", dis))


def embed_all(model_key, model_name):
    ckpt = os.path.join(OUTDIR, f"emb_{model_key}.pkl")
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            done = pickle.load(f)
        print(f"[{model_key}] resuming with {len(done)} already embedded")

    todo = [(oid, txt) for oid, txt in opinions if oid not in done]
    total = len(opinions)
    print(f"[{model_key}] {len(todo)} to embed ({total - len(todo)} already done)")

    est = lambda t: int(len(t.split()) * 1.4) + 1
    batch, btok, flushes = [], 0, 0

    def flush():
        nonlocal batch, btok, flushes
        if not batch:
            return
        ids = [b[0] for b in batch]
        texts = [b[1] if b[1].strip() else " " for b in batch]
        for attempt in range(6):
            try:
                r = vo.embed(texts, model=model_name, input_type=None, truncation=True)
                for oid, vec in zip(ids, r.embeddings):
                    done[oid] = np.asarray(vec, dtype=np.float32)
                break
            except Exception as e:
                wait = min(60, 2 ** attempt)
                print(f"  [{model_key}] error: {e} -- retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"[{model_key}] too many retries; stopping")
        flushes += 1
        batch, btok = [], 0
        if flushes % 25 == 0:
            with open(ckpt, "wb") as f:
                pickle.dump(done, f)
            print(f"  [{model_key}] checkpoint: {len(done)}/{total} embedded")

    for oid, txt in todo:
        t = est(txt)
        if t >= TOKEN_BUDGET:          # a single very long opinion: send it alone
            flush()
            batch = [(oid, txt)]; btok = t
            flush()
            continue
        if btok + t > TOKEN_BUDGET or len(batch) >= ITEM_CAP:
            flush()
        batch.append((oid, txt)); btok += t
    flush()

    with open(ckpt, "wb") as f:
        pickle.dump(done, f)
    print(f"[{model_key}] complete: {len(done)}/{total}")
    return done


def cosdist(a, b):
    if a is None or b is None:
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - float(np.dot(a, b) / (na * nb))


# Phase 1: embed everything with both models (resumable)
emb = {key: embed_all(key, name) for key, name in MODELS.items()}

# Phase 2: majority-dissent cosine distance per pair
rows = []
for i, row in df.iterrows():
    rec = {
        "pair_id": i,
        "case_key": row["case_key"],
        "dissent_ind": row.get("dissent_ind", np.nan),
    }
    for key in MODELS:
        rec[f"{key}_cosdist"] = cosdist(emb[key].get(f"{i}_maj"), emb[key].get(f"{i}_dis"))
    rows.append(rec)

out = pd.DataFrame(rows)
out_path = os.path.join(OUTDIR, "voyage_cosdist.csv")
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}  shape={out.shape}")
print(out[["voyage3_cosdist", "voyagelaw_cosdist"]].describe())

# pip install -U transformers torch scikit-learn pandas numpy tqdm

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel


output_dir = "results_filtered/legalbert/"
os.makedirs(output_dir, exist_ok=True)

# -------- Model and Tokenizer --------
# You can also try:
# MODEL_NAME = "nlpaueb/legal-bert-small-uncased"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Batch size for embedding chunks
batch_size = 16


# -------- Chunking using Legal-BERT tokenizer --------
def chunk_text_by_legalbert_tokens(text, max_tokens=512, overlap=50):
    """
    Split raw text into chunks using the Legal-BERT tokenizer.

    max_tokens includes [CLS] and [SEP], so the actual text-token limit is
    max_tokens - 2.
    """
    if not isinstance(text, str):
        text = ""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []
    start = 0
    text_token_limit = max_tokens - 2

    while start < len(token_ids):
        end = start + text_token_limit
        chunk_ids = token_ids[start:end]

        if len(chunk_ids) == 0:
            break

        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)

        if end >= len(token_ids):
            break

        start += text_token_limit - overlap

    return chunks


# -------- Mean pooling --------
def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings, ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_chunks_legalbert(chunks, batch_size=16):
    """
    Encode a list of text chunks using Legal-BERT.
    Returns one embedding per chunk.
    """
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]

        encoded = tokenizer(
            batch_chunks,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        chunk_embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        # Normalize chunk embeddings, similar to SentenceTransformer normalize_embeddings=True
        chunk_embeddings = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=1)

        all_embeddings.append(chunk_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def encode_long_document_legalbert(text, max_tokens=512, overlap=50, batch_size=16):
    """
    Convert one long legal opinion into one document embedding.
    """
    chunks = chunk_text_by_legalbert_tokens(
        text,
        max_tokens=max_tokens,
        overlap=overlap
    )

    if not chunks:
        chunks = [""]

    chunk_embs = encode_chunks_legalbert(chunks, batch_size=batch_size)

    # Average chunk embeddings
    doc_emb = np.mean(chunk_embs, axis=0)

    # Normalize final document embedding
    norm = np.linalg.norm(doc_emb)
    if norm > 0:
        doc_emb = doc_emb / norm

    return doc_emb.astype(np.float32)


# -------- Load data --------
pair_metadata_df = pd.read_csv("results_filtered/pair_metadata.csv")

pair_metadata_df["majority_text"] = pair_metadata_df["majority_text"].astype(str)
pair_metadata_df["dissent_text"] = pair_metadata_df["dissent_text"].astype(str)


# -------- Build mapping & pairs --------
document_mapping = []
pair_list = []
index = 0

for _, row in tqdm(pair_metadata_df.iterrows(), total=len(pair_metadata_df), desc="Preparing pairs"):
    document_mapping.append((index, row["case_key"], "majority"))
    document_mapping.append((index + 1, row["case_key"], f"dissent{row['dissent_ind']}"))
    pair_list.append((index, index + 1))
    index += 2

document_mapping_df = pd.DataFrame(
    document_mapping,
    columns=["index", "case_key", "opinion_label"]
)

document_mapping_df.to_csv(
    os.path.join(output_dir, "document_mapping.csv"),
    index=False
)


# -------- Embed documents with Legal-BERT --------
doc_embeddings = []

for _, row in tqdm(pair_metadata_df.iterrows(), total=len(pair_metadata_df), desc="Embedding docs"):
    for col in ["majority_text", "dissent_text"]:
        doc_emb = encode_long_document_legalbert(
            row[col],
            max_tokens=512,
            overlap=50,
            batch_size=batch_size
        )
        doc_embeddings.append(doc_emb)

doc_embeddings = np.vstack(doc_embeddings)

np.save(
    os.path.join(output_dir, "document_embeddings.npy"),
    doc_embeddings
)


# -------- Compute cosine similarities --------
cosine_similarities = []
cosine_metadata = []

for majority_idx, dissent_idx in tqdm(pair_list, desc="Computing Cosine Similarities"):
    maj_vec = doc_embeddings[majority_idx].reshape(1, -1)
    dis_vec = doc_embeddings[dissent_idx].reshape(1, -1)

    cos_sim_value = float(cosine_similarity(maj_vec, dis_vec)[0, 0])

    cosine_similarities.append(cos_sim_value)

    cosine_metadata.append((
        document_mapping_df.iloc[majority_idx]["case_key"],
        document_mapping_df.iloc[majority_idx]["opinion_label"],
        document_mapping_df.iloc[dissent_idx]["opinion_label"],
        cos_sim_value
    ))

cosine_similarities = np.array(cosine_similarities, dtype=np.float32)

np.save(
    os.path.join(output_dir, "cosine_similarities.npy"),
    cosine_similarities
)

cosine_similarity_df = pd.DataFrame(
    cosine_metadata,
    columns=[
        "case_key",
        "majority_opinion_label",
        "dissent_opinion_label",
        "cosine_similarity"
    ]
)

cosine_similarity_df.to_csv(
    os.path.join(output_dir, "cosine_similarity_metadata.csv"),
    index=False
)
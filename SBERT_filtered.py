# pip install -U sentence-transformers nltk scikit-learn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import tiktoken


output_dir = "results_filtered/sbert/"
os.makedirs(output_dir, exist_ok=True)

# -------- Model and Tokenizer --------
model = SentenceTransformer('all-distilroberta-v1')
tokenizer = tiktoken.get_encoding("gpt2")


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Batch size for embedding
batch_size = 32

# -------- Chunking (token-based) --------
def chunk_text_by_tokens(text, max_tokens=512, overlap=50):
    """
    Split raw text into chunks each having <= max_tokens model tokens.
    Overlap is in tokens (not words).
    """
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start += max_tokens - overlap
    return chunks

# -------- Load data --------
pair_metadata_df = pd.read_csv("results_filtered/pair_metadata.csv")
# pair_metadata_df["majority_text"] = pair_metadata_df["majority_text"].astype(str)
# pair_metadata_df["dissent_text"]  = pair_metadata_df["dissent_text"].astype(str)

# -------- Build mapping & pairs --------
document_mapping = []
pair_list = []
index = 0
for _, row in tqdm(pair_metadata_df.iterrows(), total=len(pair_metadata_df), desc="Preparing pairs"):
    document_mapping.append((index,   row["case_key"], "majority"))
    document_mapping.append((index+1, row["case_key"], f"dissent{row['dissent_ind']}"))
    pair_list.append((index, index+1))
    index += 2

document_mapping_df = pd.DataFrame(document_mapping, columns=["index", "case_key", "opinion_label"])
document_mapping_df.to_csv(os.path.join(output_dir, "document_mapping.csv"), index=False)

# -------- Embed with mean pooling --------
doc_embeddings = []
for _, row in tqdm(pair_metadata_df.iterrows(), total=len(pair_metadata_df), desc="Embedding docs"):
    for col in ["majority_text", "dissent_text"]:
        chunks = chunk_text_by_tokens(row[col], max_tokens=512, overlap=50)
        if not chunks:
            chunks = [""]
        chunk_embs = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device
        )
        doc_emb = np.mean(chunk_embs, axis=0)  # simple mean pooling
        doc_embeddings.append(doc_emb.astype(np.float32))

doc_embeddings = np.vstack(doc_embeddings)
np.save(os.path.join(output_dir, "document_embeddings.npy"), doc_embeddings)

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
np.save(os.path.join(output_dir, "cosine_similarities.npy"), cosine_similarities)

cosine_similarity_df = pd.DataFrame(
    cosine_metadata,
    columns=["case_key", "majority_opinion_label", "dissent_opinion_label", "cosine_similarity"]
)
cosine_similarity_df.to_csv(os.path.join(output_dir, "cosine_similarity_metadata.csv"), index=False)

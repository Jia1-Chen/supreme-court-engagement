import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import string
import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

output_dir = "results_filtered/doc2vec/"
os.makedirs(output_dir, exist_ok=True)


def preprocess_text(text):
    
    # Tokenize text into words
    all_words = [word.lower() for word in word_tokenize(text)]
    
    # Filter out words containing non-alphabetic characters
    word_tokens = [word for word in all_words if word.isalpha()]
    
    return word_tokens


# Load `pair_metadata.csv` (Filtered cases with majority length > 50)
pair_metadata_df = pd.read_csv("results_filtered/pair_metadata.csv")
pair_metadata_df["majority_text"] = pair_metadata_df["majority_text"].apply(preprocess_text)
pair_metadata_df["dissent_text"] = pair_metadata_df["dissent_text"].apply(preprocess_text)

# Create Corpus and Opinion Label Mapping
corpus = []
document_mapping = []
pair_list = []

index = 0
for _, row in tqdm(pair_metadata_df.iterrows(), desc="Building Corpus & Mapping Document Labels", total=len(pair_metadata_df)):
    majority_label = "majority"
    dissent_label = f"dissent{row['dissent_ind']}"

    corpus.append((index, row["majority_text"]))
    document_mapping.append((index, row["case_key"], majority_label))
    
    corpus.append((index + 1, row["dissent_text"]))
    document_mapping.append((index + 1, row["case_key"], dissent_label))

    # Store index pairs for Cosine Similarity computation
    pair_list.append((index, index + 1))
    
    index += 2

# Convert document mapping to DataFrame
document_mapping_df = pd.DataFrame(document_mapping, columns=["index", "case_key", "opinion_label"])
document_mapping_df.to_csv(os.path.join(output_dir, "document_mapping.csv"), index=False)

# Prepare Tagged Documents for Doc2Vec
tagged_corpus = [TaggedDocument(words=text, tags=[idx]) for idx, text in corpus]

# Train Doc2Vec Model
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
model.build_vocab(tagged_corpus)
model.train(tagged_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# Get document embeddings
document_vectors = np.array([model.dv[idx] for idx, _ in corpus])

# Save embeddings
np.save(os.path.join(output_dir, "document_embeddings.npy"), document_vectors)

# Compute Cosine Similarity for Each (Majority, Dissent) Pair
cosine_similarities = []
cosine_metadata = []
for majority_idx, dissent_idx in tqdm(pair_list, desc="Computing Cosine Similarities"):
    majority_vector = document_vectors[majority_idx].reshape(1, -1)
    dissent_vector = document_vectors[dissent_idx].reshape(1, -1)

    cos_sim_value = cosine_similarity(majority_vector, dissent_vector)[0, 0]
    cosine_similarities.append(cos_sim_value)

    # Store metadata for Cosine Similarity
    cosine_metadata.append((document_mapping_df.iloc[majority_idx]["case_key"], 
                            document_mapping_df.iloc[majority_idx]["opinion_label"], 
                            document_mapping_df.iloc[dissent_idx]["opinion_label"], 
                            cos_sim_value))

# Save Cosine Similarities as .npy
cosine_similarities = np.array(cosine_similarities)
np.save(os.path.join(output_dir, "cosine_similarities.npy"), cosine_similarities)

# Save Cosine Similarity Metadata to CSV
cosine_similarity_df = pd.DataFrame(cosine_metadata, columns=["case_key", "majority_opinion_label", "dissent_opinion_label", "cosine_similarity"])
cosine_similarity_df.to_csv(os.path.join(output_dir, "cosine_similarity_metadata.csv"), index=False)


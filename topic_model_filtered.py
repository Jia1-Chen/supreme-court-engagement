import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from tqdm import tqdm
from scipy.special import kl_div
import plotly.express as px
import os


for num_components in [90,95,105,110]:
    
    output_dir = f"results_filtered/topic_model_{num_components}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    
    
    def preprocess_text(text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [
            word for word in tokens 
            if word.isalnum() and 
               not any(char.isdigit() for char in word) and 
               re.fullmatch(r"[a-zA-Z]+", word) and 
               word not in stop_words
        ]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    
    def compute_kl_divergence(p, q):
        p = np.array(p)
        q = np.array(q)
        epsilon = 1e-10
        p = (p + epsilon) / np.sum(p + epsilon)
        q = (q + epsilon) / np.sum(q + epsilon)
        return np.sum(kl_div(p, q))
    
    
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
    
        corpus.append(row["majority_text"])
        document_mapping.append((index, row["case_key"], majority_label))
        
        corpus.append(row["dissent_text"])
        document_mapping.append((index + 1, row["case_key"], dissent_label))
    
        # Store index pairs for KL divergence computation
        pair_list.append((index, index + 1))
        
        index += 2
    
    # Convert document mapping to DataFrame
    document_mapping_df = pd.DataFrame(document_mapping, columns=["index", "case_key", "opinion_label"])
    document_mapping_df.to_csv(os.path.join(output_dir, "document_mapping.csv"), index=False)
    
    # Topic Modeling
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
    doc_term_matrix = vectorizer.fit_transform(corpus)
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=num_components, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get topic distributions
    topic_distributions = lda.transform(doc_term_matrix)
    
    # Get vocabulary
    vocabulary = vectorizer.get_feature_names_out()
    topic_names = [f"Topic {i}" for i in range(lda.n_components)]
    
    # Create DataFrames
    topic_word_distributions = pd.DataFrame(lda.components_, columns=vocabulary, index=topic_names)
    document_topic_distributions = pd.DataFrame(topic_distributions, columns=topic_names)
    
    # Add `case_key` and `opinion_label` columns
    document_topic_distributions.insert(0, "case_key", document_mapping_df["case_key"])
    document_topic_distributions.insert(1, "opinion_label", document_mapping_df["opinion_label"])
    
    # Save DataFrames to CSV
    topic_word_distributions.to_csv(os.path.join(output_dir, "topic_word_distributions.csv"), index=False)
    document_topic_distributions.to_csv(os.path.join(output_dir, "document_topic_distributions.csv"), index=False)
    
    # Compute KL Divergence for Each (Majority, Dissent) Pair
    kl_divergences = []
    kl_metadata = []
    for majority_idx, dissent_idx in tqdm(pair_list, desc="Computing KL Divergences"):
        majority_distribution = topic_distributions[majority_idx]
        dissent_distribution = topic_distributions[dissent_idx]
    
        kl_div_value = compute_kl_divergence(majority_distribution, dissent_distribution)
        kl_divergences.append(kl_div_value)
    
        # Store metadata for KL divergence
        kl_metadata.append((document_mapping_df.iloc[majority_idx]["case_key"], 
                            document_mapping_df.iloc[majority_idx]["opinion_label"], 
                            document_mapping_df.iloc[dissent_idx]["opinion_label"], 
                            kl_div_value))
    
    # Save KL Divergences as .npy
    kl_divergences = np.array(kl_divergences)
    np.save(os.path.join(output_dir, "kl_divergences.npy"), kl_divergences)
    
    # Save KL Divergence Metadata to CSV
    kl_divergence_df = pd.DataFrame(kl_metadata, columns=["case_key", "majority_opinion_label", "dissent_opinion_label", "kl_divergence"])
    kl_divergence_df.to_csv(os.path.join(output_dir, "kl_divergence_metadata.csv"), index=False)
    
    # Save topic distributions as .npy
    np.save(os.path.join(output_dir, "topic_distributions.npy"), topic_distributions)
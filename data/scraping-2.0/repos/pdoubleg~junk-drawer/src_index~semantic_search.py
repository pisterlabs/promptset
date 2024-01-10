import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import faiss
import nltk
from typing import List, Tuple, Dict
from langchain.embeddings import OpenAIEmbeddings


class SemanticSearch:
    def __init__(self, df: pd.DataFrame, embedding_col_name: str = "embeddings"):
        self.df = df
        self.embedding_col_name = embedding_col_name
        load_dotenv()
        self.embedding_model = OpenAIEmbeddings()


    @staticmethod
    def preprocess_text(text: str):
        text = text.lower()
        # remove stopwords from the query
        stopwords = set(nltk.corpus.stopwords.words("english"))
        # Example of adding words to the stopwords list
        stopwords.update(["please", "review"])
        text = " ".join([word for word in text.split() if word not in stopwords])
        return text

    
    def encode_string(self, text: str) -> np.ndarray:
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        # Encode the text
        embedding = self.embedding_model.embed_query(preprocessed_text)
        return np.array(embedding)
    
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        if embeddings.shape[0] == 1:
            embeddings = np.transpose(embeddings)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        return normalized_embeddings
    
    
    def check_and_normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        # Calculate the L2 norm for each embedding
        norms = np.linalg.norm(embeddings, axis=1)

        # Check if the norms are close to 1 (with a small tolerance)
        if not np.allclose(norms, 1, atol=1e-6):
            print("Embeddings are not normalized, normalizing now...")
            embeddings = self.normalize_embeddings(embeddings)

        return embeddings
    
    
    def build_faiss_index(self, embeddings: np.ndarray, use_cosine_similarity: bool = False) -> faiss.Index:
        if use_cosine_similarity:
            # Check and normalize the embeddings if needed
            embeddings = self.check_and_normalize_embeddings(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])

        index.add(embeddings.astype('float32'))
        return index

       
    def search_faiss_index(self, index: faiss.Index, 
                            embedding: np.ndarray,
                            top_n: int,
                            use_cosine_similarity: bool,
                            similarity_threshold: float) -> Tuple[List[int], np.ndarray]:
        
        distances, indices = index.search(embedding.reshape(1, -1).astype('float32'), top_n + 1)
        
        if use_cosine_similarity:
            similarity_scores = distances.flatten()
        else:
            similarity_scores = 1 - distances.flatten()
        
        # Exclude results that are too similar
        indices = indices.flatten()[similarity_scores < similarity_threshold]
        similarity_scores = similarity_scores[similarity_scores < similarity_threshold]
        
        return indices[:top_n], similarity_scores[:top_n]


    @staticmethod
    def filter_results(df: pd.DataFrame, indices: List[int], filter_criteria: dict) -> pd.DataFrame:
        filtered_df = df.iloc[indices]
        for key, value in filter_criteria.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df
    
    
    def query_similar_documents(self, text: str, top_n: int, filter_criteria: dict, use_cosine_similarity: bool = False, similarity_threshold: float = 0.99) -> pd.DataFrame:
        query_embedding = self.encode_string(text)

        if filter_criteria is not None:
            filtered_df = self.df.copy()
            for key, value in filter_criteria.items():
                filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = self.df.copy()
            
        filtered_embeddings = np.vstack(filtered_df[self.embedding_col_name].values)

        index_ = self.build_faiss_index(filtered_embeddings, use_cosine_similarity)
        indices, sim_scores = self.search_faiss_index(index_, query_embedding, top_n, use_cosine_similarity, similarity_threshold)
        results_df = filtered_df.iloc[indices].copy()
        # Add 'similarity scores' to the DataFrame
        results_df['sim_score'] = sim_scores
        
        return results_df
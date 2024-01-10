import os
import sys

import sys

import openai
import whisper

import numpy as np
import pandas as pd
import nltk
import tiktoken

import pickle
import time
from tqdm import tqdm

from pytube import YouTube

from gpt3summarizer import GPT3Summarizer

class EmbeddingEngine():
    def __init__(self, openai_key):
        openai.api_key  = openai_key
        self.embedding_model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str):
        if text is not None:
            try:
                result = openai.Embedding.create(
                  model=self.embedding_model,
                  input=text
                )
                return result["data"][0]["embedding"]
            except:
                return "Rate Limit Reached"
        else:
            return None
    
    def compute_doc_embeddings(self, df: pd.DataFrame, label:str):
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        embeddings_dict = {}

        for idx, r in tqdm(df.iterrows()):
            embedding = self.get_embedding(r[label])

            # Delay 60s if rate limit reached
            if type(embedding) == str:
                time.sleep(60)
                embedding = self.get_embedding(r[label])

            embeddings_dict[idx] = embedding

        return embeddings_dict
    
    def vector_similarity(self, x, y):
        """
        Returns the similarity between two vectors.

        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        if x is not None and y is not None:
            return np.dot(np.array(x), np.array(y))
        else:
            return -1
    
    def order_document_sections_by_query_similarity(self, query, contexts):
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 

        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities
    
class ChatEngine():
    def __init__(self, openai_key):
        self.openai_key = openai_key
        openai.api_key  = self.openai_key
    
    def construct_context(self, question: str, context_embeddings: dict, df: pd.DataFrame, label:str, 
                          MAX_SECTION_LEN = 1000, SEPARATOR = "\n* ", ENCODING = "gpt2", debug=False):
                      
        def num_tokens_from_string(string: str, encoding_name: str) -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.get_encoding(encoding_name)
            num_tokens = len(encoding.encode(string))
            
            return num_tokens
        
        embed_engine = EmbeddingEngine(openai_key = self.openai_key)
        most_relevant_document_sections = embed_engine.order_document_sections_by_query_similarity(question, context_embeddings)
    
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        encoding = tiktoken.get_encoding(ENCODING)
        separator_len = len(encoding.encode(SEPARATOR))

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index][label]

            if document_section is None:
                continue

            chosen_sections_len += num_tokens_from_string(document_section, ENCODING) + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        if debug:
            print(f"Selected {len(chosen_sections)} document sections:")
            print("\n".join(chosen_sections_indexes))
    
        return chosen_sections
                      
    def construct_prompt(self, question_prompt, document_embeddings, document_df, max_sentences=10):
        
        context = self.construct_context(question_prompt, document_embeddings, document_df, label="content")

        prompt = f"Instructions:\nAnswer the question as truthfully as possible using the conversation snippets of a podcast. "
        prompt += f"If the answer is not contained within the text below, say 'I dont know.'. "
        prompt += f"Limit your answer to a paragraph of {max_sentences} sentences. "
        prompt += f"\n\nContext: {''.join(context)}"
        prompt += f"\n\nQuestion: {question_prompt}"
        prompt += f"\n\nAnswer: "
        
        self.prompt = prompt
    
    def call(self):
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content": self.prompt},
                    ]
            )
        choices = response.choices
        if choices:
            content = choices[0].message.content
            total_tokens = response.usage.total_tokens
        else:
            print(response)
            Exception("No choices returned from GPT-3 API using model 'gpt-3.5-turbo'")
        
        return content
    

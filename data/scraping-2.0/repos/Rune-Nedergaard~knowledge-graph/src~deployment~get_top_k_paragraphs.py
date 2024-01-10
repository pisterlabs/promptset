import argparse
import sys
import os
import openai
import logging
# Set the logging level for all loggers to ERROR before importing the classes
logging.getLogger().setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from src.models.bert_embed import BertEmbed
from api_secrets import API_KEY
#from src.models.semantic_search_function import get_similar_paragraphs
from src.models.semantic_search_two_towers import get_similar_paragraphs
from src.models.bert_rerank import BertRerank
from src.models.reranker_function import rerank_paragraphs
import numpy as np
from src.features.two_towers_fine_tune_multiplenegatives import *
from tqdm import tqdm

openai.api_key = API_KEY

def get_top_paragraphs(questions, use_reranker=True):
    # Load the fine-tuned re-ranker model
    fine_tuned_model_path = 'models/fine_tuned_model/check_highbatch2_model.pth'

    # Load the fine-tuned re-ranker model
    reranker = BertRerank(model_path=fine_tuned_model_path)

    all_results = []

    for question in tqdm(questions, desc="Getting top paragraphs"):
        # Perform semantic search to get the top 1000 paragraphs
        similar_paragraphs, similar_file_indices = get_similar_paragraphs(question, k=100)

        if use_reranker:
            # Rerank the paragraphs using the rerank_paragraphs function
            ranked_paragraphs = rerank_paragraphs(question, similar_file_indices, similar_paragraphs, reranker)
        else:
            # If not using reranker, take the top 8 paragraphs directly from similar_paragraphs
            ranked_paragraphs = zip(similar_paragraphs[:8], [None]*8, similar_file_indices[:8], range(8))

        all_info_paragraphs = []
        seen = set()  # Create an empty set to keep track of seen elements
        for context_paragraph, score, filename, paragraph_index in ranked_paragraphs:
            if context_paragraph not in seen:  # Check if the element has not been seen before
                all_info_paragraphs.append((context_paragraph, score, filename, paragraph_index))
                seen.add(context_paragraph)  # Add the element to the seen setl, then we remove duplicates

        all_results.append(all_info_paragraphs[:8])

    return all_results



if __name__ == "__main__":
    print("main was activated, this probably shouldn't happen")
    #main("Producerer Danmark flere vindmøller i dag end for 10 år siden?")

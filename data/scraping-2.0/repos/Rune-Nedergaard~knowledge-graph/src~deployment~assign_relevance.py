"""
This script takes a dict of rephrased questions and their semantic search results, then it does the following:
1. It creates a gpt3.5 prompt template that asks for the relevance of a paragraph to a question and inputs the question and paragraph
2. It extracts the results and creates triplets of (question, paragraph, relevance)
3. It saves these to a dataformat that a BERT model can read and fine tune
"""
import openai
from api_secrets import API_KEY
import os
import re
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import multiprocessing
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.deployment.get_top_k_paragraphs import get_top_paragraphs
from src.models.semantic_search_two_towers import get_similar_paragraphs
from src.models.bert_rerank import BertRerank
from src.models.reranker_function import rerank_paragraphs
import numpy as np
from src.features.two_towers_fine_tune_multiplenegatives import *
openai.api_key = API_KEY
failed_files = []

output_folder = 'data/reranker_triplets'

def process_question(question, list_of_paragraphs):
    retry = 0
    template = ''''Input spørgsmål: {question}
                    
Tekststykke 1: {list_of_paragraphs[0]}
                    
Tekststykke 2: {list_of_paragraphs[1]}
                    
Tekststykke 3: {list_of_paragraphs[2]}
                    
Tekststykke 4: {list_of_paragraphs[3]}

Tekststykke 5: {list_of_paragraphs[4]}
'''
    REQUEST = template.format(question=question, list_of_paragraphs=list_of_paragraphs)
    #count the number of tokens in the request
  
    while retry < 1:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": '''Du vurderer hvor relevante 5 forskellige tekststykker er i forhold til et givent input spørgsmål. Hver paragraf vurderes på en skala fra 0-1, hvor 0 er "slet ikke relevant" og 1 er "meget relevant". Det er sandsynligt at ingen er relevante, så vær hård i din vurdering og giv kun en høj score, hvis det faktisk er relevant. Svar uden yderligere forklaring med et decimaltal per paragraf, således:
                    
                    Tekststykke 1: 0.1
                    Tekststykke 2: 0.6
                    Tekststykke 3: ....'''},

                    {"role": "user", "content": REQUEST}
                ],
                temperature=0,
                max_tokens=100,
            )

            response_text = response['choices'][0]['message']['content']
            relevance_scores = re.findall(r'Tekststykke \d+: (\d(?:\.\d+)?)', response_text)


            return relevance_scores

        except Exception as e:
            print(e)
            retry += 1
            if retry == 2:
                print(f"Failed to process question {id}")
                failed_files.append(id)
                break

if __name__ == '__main__':



    #opem all the txt files in the data/testset_questions folder and read them into a list as strigns

    #get the list of files
    files = glob.glob('data/testset_questions/*.txt')
    #read the files into a list
    questions = []
    for file in files:
        with open(file, 'r', encoding='iso-8859-1', errors='replace') as f:
            questions.append(f.read())
    
    results = get_top_paragraphs(questions, use_reranker=False)

    relevance_output_folder = "data/relevance_scores_no_reranker"
    paragraph_output_folder = "data/found_paragraphs_no_reranker"
    os.makedirs(relevance_output_folder, exist_ok=True)
    os.makedirs(paragraph_output_folder, exist_ok=True)

    for idx, question in enumerate(questions):
        top_paragraphs = results[idx][:5]
        list_of_paragraphs = [paragraph[0] for paragraph in top_paragraphs]

        relevance_scores = process_question(question, list_of_paragraphs)

        if relevance_scores:
            # Save relevance scores
            with open(os.path.join(relevance_output_folder, f"{idx}.txt"), 'w') as f:
                for i, score in enumerate(relevance_scores):
                    f.write(f"Tekststykke {i + 1}: {score}\n")

            # Save paragraphs
            with open(os.path.join(paragraph_output_folder, f"{idx}.txt"), 'w', encoding='iso-8859-1', errors='replace') as f:
                for paragraph in list_of_paragraphs:
                    f.write(paragraph)
                    f.write("\n\n")
        else:
            print(f"Failed to process question {idx}")
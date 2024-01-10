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
#from src.features.qa_search_danlp import find_question_answer_pairs
from src.features.old_qa_search import find_question_answer_pairs
#from src.features.qa_search import find_question_answer_pairs, initialize_qa_search

openai.api_key = API_KEY
failed_files = []

output_folder = 'data/reranker_triplets'

def read_qa_pairs(filename):
    qa_pairs = []
    with open(filename, 'r', encoding='iso-8859-1', errors='replace') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            qa_pairs.append((question, answer))
    return qa_pairs

def process_question(question, qa_pairs):
    retry = 0
    try:
        template = '''Input spørgsmål: {question}

        Par 1: {pair1}

        Par 2: {pair2}

        Par 3: {pair3}

        Par 4: {pair4}

        Par 5: {pair5}
        '''

        REQUEST = template.format(question=question, pair1=qa_pairs[0], pair2=qa_pairs[1], pair3=qa_pairs[2], pair4=qa_pairs[3], pair5=qa_pairs[4])
    except:
        print('Error in formatting question: ', question)
        pass
    #count the number of tokens in the request
  
    while retry < 1:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": '''Du vurderer hvor relevante 5 forskellige spørgsmål-svar par er i forhold til et givent input spørgsmål. Hvert spørgsmål-svar par vurderes på en skala fra 0-1, hvor 0 er "slet ikke relevant" og 1 er "meget relevant". Det er sandsynligt at ingen er relevante, så vær hård i din vurdering og giv kun en høj score, hvis det faktisk er relevant. Svar uden yderligere forklaring med et decimaltal per spørgsmål-svar par, således:

                            Par 1: 0.1
                            Par 2: 0.6
                            Par 3: ....'''},

                    {"role": "user", "content": REQUEST}
                ],
                temperature=0,
                max_tokens=100,
            )

            response_text = response['choices'][0]['message']['content']
            relevance_scores = re.findall(r'Par \d+: (\d(?:\.\d+)?)', response_text)


            return relevance_scores

        except Exception as e:
            print(e)
            retry += 1
            if retry == 2:
                print(f"Failed to process question {id}")
                failed_files.append(id)
                break

if __name__ == '__main__':

    qa_pairs = read_qa_pairs('data/random.txt')

    # Get the list of files
    files = glob.glob('data/testset_questions/*.txt')
    # Read the files into a list
    questions = []
    for file in files:
        with open(file, 'r', encoding='iso-8859-1', errors='replace') as f:
            questions.append(f.read())


    results = [qa_pairs for _ in questions]

    relevance_output_folder = "data/relevance_random"
    paragraph_output_folder = "data/found_paragraphs_random"
    os.makedirs(relevance_output_folder, exist_ok=True)
    os.makedirs(paragraph_output_folder, exist_ok=True)

    for idx, question in enumerate(questions):
        top_qa_pairs = results[idx][:5]

        relevance_scores = process_question(question, top_qa_pairs)

        if relevance_scores:
            # Save relevance scores
            with open(os.path.join(relevance_output_folder, f"{idx}.txt"), 'w') as f:
                for i, score in enumerate(relevance_scores):
                    f.write(f"Par {i + 1}: {score}\n")

            # Save paragraphs
            with open(os.path.join(paragraph_output_folder, f"{idx}.txt"), 'w', encoding='iso-8859-1', errors='replace') as f:
                for qa_pair in top_qa_pairs:
                    f.write(qa_pair[0])
                    f.write("\n\n")
        else:
            print(f"Failed to process question {idx}")

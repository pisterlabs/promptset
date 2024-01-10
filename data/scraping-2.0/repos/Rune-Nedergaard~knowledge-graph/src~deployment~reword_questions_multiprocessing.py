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

openai.api_key = API_KEY
failed_files = []

output_folder = 'data/questions_rephrased'

questions = pd.read_pickle('data/raw/random_subset3.pkl')# doing another 20.0000 + now another 35000
ids = [question[0] for question in questions]
question_text = [question[1] for question in questions]
#remove space at the begnning of the string if there is one
question_text = [re.sub(r'^\s+', '', question) for question in question_text]
#remove space at the end of the string if there is one
question_text = [re.sub(r'\s+$', '', question) for question in question_text]




zipped = zip(question_text, ids)
zipped = list(zipped)


def process_question(text, id):
    retry = 0
    while retry < 2:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Du omskriver spørgsmål til mere generelle spørgsmål uden mange detaljer. Den omskrevne tekst skal minde om noget en normal person ville bruge til at søge efter information online og skal ikke være adresseret til nogen bestemt."},
                    {"role": "user", "content": text},
                ],
                temperature=0.5,
                max_tokens=100,
            )

            response_text = response['choices'][0]['message']['content']

            with open(os.path.join(output_folder, f'{id}.txt'), 'w') as f:
                f.write(response_text)
            break
        except Exception as e:
            print(e)
            retry += 1
            if retry == 2:
                print(f"Failed to process question {id}")
                failed_files.append(id)
                break

if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    results = pool.starmap(process_question, zipped)
    pool.close()
    pool.join()

    with open('data/failed_question_rephrased3.pkl', 'wb') as f:
        pickle.dump(failed_files, f)



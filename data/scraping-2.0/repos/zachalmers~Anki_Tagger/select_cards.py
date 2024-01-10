import pandas as pd
import numpy as np
import re, sys, csv, os
import openai
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
from openai.error import RateLimitError, APIError, ServiceUnavailableError
import time, requests

MAX_POOR_MATCH_RUN = 10
MAX_TOKENS_PER_OBJ = 5000

def set_api_key():
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    except KeyError:
        print("Set your OpenAI API key as an environment variable named 'OPENAI_API_KEY' eg In terminal: export OPENAI_API_KEY=your-api-key")

def handle_api_error(func):
    def wrapper(*args, **kwargs):
        t=5
        while True:
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIError, ServiceUnavailableError):
                print(f'API Error. Waiting {t}s before retrying.')
                time.sleep(t)  # wait for 10 seconds before retrying
                t+=5
    return wrapper

def convert_to_np_array(s):
    return np.fromstring(s.strip("[]"), sep=",")

def load_emb(path):

    # Specify the data types for columns 0, 1, and 2
    column_dtypes = {0: str, 1: str, 2: int}

    # Read CSV file and interpret column types
    df = pd.read_csv(
        path,
        dtype=column_dtypes,
        converters={3: convert_to_np_array})

    return df

def vs(x, y):
    return np.dot(np.array(x), np.array(y))

def construct_prompt(obj,card):

    prompt = f"Rate how relevant the anki card is to the learning question on a scale from 0 to 100 and return the score.\n\
    Learning question: {obj}\n\
    Anki card: {card}"

    formatted_prompt = [{"role": "system", "content": "You are an assistant that precisely follows instructions."},
                        {"role": "user", "content": prompt}]

    return formatted_prompt

def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = list(enc.encode(text))
    return len(tokens)

def tokens_in_prompt(formatted_prompt):
    formatted_prompt_str = ""
    for message in formatted_prompt:
        formatted_prompt_str += message["content"] + " "
    return count_tokens(formatted_prompt_str)

@handle_api_error
def rate_card_for_obj(prompt, temperature=1):

    # Calculate the remaining tokens for the response

    remaining_tokens = 4096 - tokens_in_prompt(prompt) - 20

    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the gpt-3.5-turbo engine
        messages=prompt,
        max_tokens=remaining_tokens,  # Set the remaining tokens as the maximum for the response
        n=1,
        stop=None,
        temperature=temperature)

    string_return = completions['choices'][0]['message']['content'].strip()
    return string_return.replace('\n',' ')

def clean_reply(s):

    matches = re.search(r'Score: (\d{1,3})', s)

    if matches:
        score = matches.group(1)
        return int(score)

    else:
        matches = re.findall(r'\b([0-9][0-9]?|100)\b', s)
        if matches:
            numbers = [int(num) for num in matches]
            return min(numbers)
        else:
            return "NA"

def main(emb_path,obj_path):

    output_prefix = os.path.basename(obj_path).replace("_learning_objectives.csv",'')

    # load previous progress if exists
    last_processed_index = -1
    progress_file = f"{output_prefix}_progress.csv"

    if os.path.exists(progress_file):
        last_progress_df = pd.read_csv(progress_file)
        if not last_progress_df.empty:
            last_processed_index = last_progress_df.iloc[-1][0]

    emb_df = load_emb(emb_path)
    obj_df = load_emb(obj_path)

    with open(f'{output_prefix}_cards.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if last_processed_index == -1:  # if there's no previous progress
            csv_writer.writerow(['guid','card','tag','cosine_sim','gpt_reply','score','objective'])

        for obj_index,obj_row in obj_df.iterrows():

            if obj_index <= last_processed_index:
                continue  # skip if the row has already been processed

            print(f"Processing objective {obj_index}")
            tag = obj_row['name']
            obj = obj_row['learning_objective']
            tokens = obj_row['tokens']
            obj_emb = obj_row['emb']

            emb_df["cosine_sim"] = emb_df.emb.apply(lambda x: vs(obj_emb,x))
            emb_df.sort_values(by='cosine_sim', ascending=False, inplace=True)

            poor_match_run_count = 0
            tokens_used = 0

            for index,emb_row in emb_df.iterrows():

                if poor_match_run_count > MAX_POOR_MATCH_RUN or tokens_used > MAX_TOKENS_PER_OBJ:
                    break

                guid = emb_row['guid']
                card = emb_row['card']
                cosine_sim = emb_row["cosine_sim"]
                gpt_reply = "NA"
                score = "NA"

                prompt = construct_prompt(obj,card)
                tokens_used += tokens_in_prompt(prompt)

                #try with progressively more creative juice
                temp = 0
                while score == "NA" and temp <= 1:
                    gpt_reply = rate_card_for_obj(prompt, temperature=temp)
                    score = clean_reply(gpt_reply)
                    temp += 0.25

                csv_writer.writerow([guid,card,tag,cosine_sim,gpt_reply,score,obj])
                if score > 50:
                    poor_match_run_count=0
                else:
                    poor_match_run_count+=1

            with open(progress_file, 'a', newline='', encoding='utf-8') as progress_csvfile:
                progress_csv_writer = csv.writer(progress_csvfile)
                progress_csv_writer.writerow([obj_index])

if __name__ == "__main__":
    set_api_key()
    if len(sys.argv) != 3:
        print("Usage: select_cards.py <deck_embeding> <learning_objectives>")
        sys.exit(1)
    emb_path = sys.argv[1]
    obj_path = sys.argv[2]
    main(emb_path,obj_path)

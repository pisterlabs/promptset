import pandas as pd
import openai
import concurrent.futures
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import time
import os
file_path = 'responses.txt'
openai.api_key = open("key.txt", "r").read().strip("\n")

# Define the rate limit parameters
subreddit = "AskCulinary"
RATE_LIMIT = 5  # Number of calls allowed per second
RATE_LIMIT_PERIOD = 60  # Time period in seconds


def REDDIT_PROMPT(question, context):
    return f"""
Question: {question}
Context: {context}
Answer:
"""

def get_done_set():
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing JSON file
        with open(file_path, 'r') as file:
            answers = set(file.read().split())
        print("File exists. Data read successfully.")
    else:
        # Create a new JSON file
        answers = set()
        with open(file_path, 'w') as file:
            file.write('')
        print("File created. Data initialized.")
    return answers

answers = get_done_set()

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=RATE_LIMIT_PERIOD)
def get_completion(row):
    q_id = row['id']
    question = REDDIT_PROMPT(row['title'], row['text'])
    if q_id not in answers:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
                messages=[{"role":"system","content":f"You are now a helpful Reddit user from {subreddit} who helps other people by answering their question concisely and with a friendly and gentle tone. You will use the question and context to answer the question. I repeat, be very concise."},{"role": "user", "content": question}]
            )
            # Update the dataframe 'chatgpt-3.5-turbo' column with the answer
            data['chatgpt-3.5-turbo'] = data['chatgpt-3.5-turbo'].where(data['id'] != q_id, completion["choices"][0]["message"]["content"])
            #data.loc[data['id'] == q_id, 'chatgpt-3.5-turbo'] = completion["choices"][0]["message"]["content"]
            with open(file_path,'a') as f:
                f.write(q_id + '\n')
            with open('log.txt','a') as f:
                f.write(q_id + '\n')
                f.write("Question: " + question + '\n')
                f.write("Answer: " + completion["choices"][0]["message"]["content"] + '\n')
                f.write("--------------------------------------------------\n")
        except openai.error.RateLimitError as e:
            with open('log.txt','a') as f:
                f.write(q_id + '\n')
                f.write("Sleeping for 60" + '\n')
                f.write("--------------------------------------------------\n")
            time.sleep(60)
            get_completion(question)
        except Exception as e:
            with open('log.txt','a') as f:
                f.write(q_id + '\n')
                f.write("Exception: " + str(e) + '\n')
                f.write("--------------------------------------------------\n")
            print(f"Unkown Exception - {e}")
        finally:
            return
    else:
        with open('log.txt','a') as f:
            f.write(q_id + '\n')
            f.write("Skipping" + '\n')
            f.write("--------------------------------------------------\n")
        return

def get_answers(df):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        rows = []
        for idx, row in df.iterrows():
            rows.append(row)
        with tqdm(total=len(df)) as pbar:
            for result in executor.map(get_completion, rows):
                pbar.update(1)
                data.to_csv('data_AskCulinary.csv', index=False)
data = pd.read_csv('data_AskCulinary.csv')
get_answers(data)
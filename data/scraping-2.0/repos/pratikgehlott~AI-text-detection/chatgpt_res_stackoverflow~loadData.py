import pandas as pd
import openai
import concurrent.futures
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import time
import os
from uuid import uuid4

openai.api_key = open("key.txt", "r").read().strip("\n")

# Define the rate limit parameters
RATE_LIMIT = 5  # Number of calls allowed per second
RATE_LIMIT_PERIOD = 60  # Time period in seconds



file_path = 'merged_stackoverflow_35_54'

df = pd.read_pickle(file_path + '.pkl')
column_to_check = 'post_answer'  # Replace 'column_name' with the actual column name
value_to_drop = ""  # Replace 'value_to_drop' with the specific value you want to drop

df_new = df[df['post_answer'] != value_to_drop]
df_new['id'] = [str(uuid4()) for _ in range(len(df_new.index))]
df_new['chatgpt-3.5-turbo'] = '[EMPTY]'



avg_len = df_new['post_question'].str.split().str.len().mean()
print(f"Number of Posts: {len(df_new)}\nAverage Length of Posts: {avg_len}")




def STACKOVERFLOW_PROMPT(question):
    return f"""
Question: {question}
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
    question = STACKOVERFLOW_PROMPT(row['post_question']) 
    if q_id not in answers:
        try:
            completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "system", "content": f"You are now a helpful Stackoverflow user who helps other people by "
                                                f"answering their question succinctly and with a friendly and gentle "
                                                f"tone. You will use the question to answer the question "
                                                f"in a comprehensive manner. If you do not have enough information on "
                                                f"the question asked, you will return with some blog or documentation "
                                                f"references for the user."}, {"role": "user", "content": question}]
            )
            # Update the dataframe 'chatgpt-3.5-turbo' column with the answer
            df_new['chatgpt-3.5-turbo'] = df_new['chatgpt-3.5-turbo'].where(df_new['id'] != q_id, completion["choices"][0]["message"]["content"])
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
    


# def get_answers(df):
#     # Define your chat prompts
#     prompts = [STACKOVERFLOW_PROMPT(question) for question in df['post_question'].values]
#     print(f"LEN = {len(prompts)}")
#     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#         results = []
#         with tqdm(total=len(prompts)) as pbar:
#             for result in executor.map(get_completion, prompts):
#                 results.append(result)
#                 pbar.update(1)
#     return results


# df_new['gpt-3.5-turbo'] = get_answers(df_new)
# print(df_new.head(2))
# df_new.to_pickle(file_path + '_Answers.pkl', index=False)


def get_answers(df):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        rows = []
        for idx, row in df.iterrows():
            rows.append(row)
        with tqdm(total=len(df)) as pbar:
            for result in executor.map(get_completion, rows):
                pbar.update(1)
                df_new.to_csv('merged_stackoverflow_35_54_Answers.csv', index=False)




get_answers(df_new)


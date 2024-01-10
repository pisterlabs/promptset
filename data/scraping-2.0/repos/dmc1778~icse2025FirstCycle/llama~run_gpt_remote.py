
import pandas as pd
import tiktoken
from openai import OpenAI
import os
from dotenv import load_dotenv
import csv

load_dotenv()
client = OpenAI(
    api_key=os.environ.get(".env")
)

def create_prompt(post):
    prompt_ = f"""
    You are an advanced stackoverflow post bot analyzer. 
    Your duty is extract at most three meaningful keywords related to GPU errors in the given stackoverflow post wrapped by ####.
    Stackoverflow post: ####{post}####

    REMEMBER: Do not explain the bug, just extract some keywords. 
    REMEMBER: If you can not extract any related keywords, just skip generate any response. 

    Please generate the keywords as the following format:

    Keywords: keyword1, keyword2, keyword3.
    """
    return prompt_

def completions_with_backoff(prompt, model='gpt-3.5-turbo-1106'):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response

def get_token_count(string):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")

    num_tokens = len(encoding.encode(string))

    return num_tokens

def write_csv(data, target, stage=3):
    if not os.path.exists(f'output/keywords/{target}'):
        os.makedirs(f'output/keywords/{target}')

    file_path = f"output/keywords/keywords_{target}.csv"

    with open(file_path, 'a', newline='\n', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data)

def parser():
    data = pd.read_csv('output/posts/stage2/stage2_device.csv', sep=',', encoding='utf-8')
    for idx, row in data.iterrows():
        print(f"I am processing the record {idx}/{len(data)}")
        prompt_ = create_prompt(row.iloc[1])
        t_count = get_token_count(prompt_)
        if t_count <= 4097:
            conversations = completions_with_backoff(prompt_)
            response = conversations.choices[0].message.content
            write_csv([response],'device')
            



if __name__ == '__main__':
    parser()
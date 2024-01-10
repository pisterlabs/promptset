import numpy as np
from langchain.llms import OpenAI
from langchain import LLMChain
import os
os.environ["OPENAI_API_KEY"] = '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'
from langchain.llms import OpenAIChat
import tiktoken

from question import simple_question as question
from time import sleep
from tqdm import tqdm
import csv

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == '__main__':
    data_file = 'samples_2022.pkl'
    data = np.load(data_file, allow_pickle=True)

    openaichat = OpenAIChat(model_name="gpt-3.5-turbo", temperature=0.)

    with open('answers_2022.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'prompt_token_count','news_count', 'target_trend', 'target_volatility', 'target_active_level', 'company', 'date', 'num_token', 'answer'])

    token_total = 1
    for prompt, news_count, target, company, date in tqdm(data):
        prompt = prompt.replace('\t', '')
        num_token = num_tokens_from_string(prompt+question, "cl100k_base")
        deduce_count = 0
        while num_token > 4096:
            deduced = 100 * deduce_count
            prompt = prompt[:4096-deduced]
            num_token = num_tokens_from_string(prompt+question, "cl100k_base")
            deduce_count += 1
        token_total += num_token
        # print(num_token, token_total)
        answer = openaichat(prompt+question)
        with open('answers_2022.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([prompt, num_token, news_count, target['trend'], target['volatility'], target['active_level'], company, date, num_token, answer])
        sleep(6)

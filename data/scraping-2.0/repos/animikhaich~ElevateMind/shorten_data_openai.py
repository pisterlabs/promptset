import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import argparse
import os
import mapply
import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()

argparser.add_argument('--data_path', type=str, default='data/conversations.csv')
argparser.add_argument('--api_key', type=str, default=os.environ.get('OPENAI_API_KEY'))

args = argparser.parse_args()

assert args.api_key

client = OpenAI(
    api_key=args.api_key,
)

api_call_counter = 0

def chatGPTSummarize(context):
    global api_call_counter
    api_call_counter += 1

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are an expert who will summarize in less than 1000 characters the following conversation between a 'usr' and a 'sys' without losing information and without adding new information."},
            {"role": "user", "content": context},
        ]
    )
    
    return response.choices[0].message.content

def summarizeIfLongerThan(context, max_len=3000):
    try:
        if len(context) > max_len:
            return chatGPTSummarize(context)
        else:
            return context
    except Exception as e:
        print(e)
        return context

df = pd.read_csv(args.data_path)
df = df.where(df.notna(), "") 

mapply.init(n_workers=-1)
df['context'].mapply(summarizeIfLongerThan)


print(f"API calls: {api_call_counter}")
df.to_csv('data/conversations_summarized.csv', index=False)

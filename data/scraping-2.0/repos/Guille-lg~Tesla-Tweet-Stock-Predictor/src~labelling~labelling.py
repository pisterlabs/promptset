import pandas as pd
import os
from dotenv import load_dotenv
import openai
import requests
import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()
system_prompt = "You are a data scientist labelling tweets. Given a tweet, classify it into one of the following categories: Tesla Products, Customer Experience, Performance & Innovation, Financial News, Environmental Impact, Industry News, Charging Infrastructure. If the tweet does not clearly fit into any of these categories or is not relevant, classify it as 'Not relevant'. Example: Tweet: Tesla's new Model S Plaid is the fastest production car ever made! Category: Tesla Products. Example: Tweet: I'm having a terrible time with Tesla customer service. They've been ignoring my emails for weeks. Category: Customer Experience. I want you to be very rigorous with the labelling. You have to respond with the following phrase: Category: [response]"
label_map = {
    'Tesla Products': 0,
    'Customer Experience': 1,
    'Performance & Innovation': 2,
    'Financial News': 3,
    'Environmental Impact': 4,
    'Industry News': 5,
    'Charging Infrastructure': 6,
}

headers = {
    'Authorization' : 'Bearer ' + os.getenv('OPENAI_API_KEY'),
    'Content-Type' : 'application/json',
}

class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.done = 0

    def increment(self):
        self.done = self.done + 1

    def __repr__(self):
        return f"Done runs {self.done}/{self.total}."


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), before_sleep=print, retry_error_callback=lambda _: None)
async def get_completion(row, session, semaphore, progress_log):
    async with semaphore:

        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": system_prompt + ' The tweet I want you to label is the following: ' + row[2]}
                ],
            "temperature": 0,
            "max_tokens": 7,
        }) as resp:

            response_json = await resp.json()

            progress_log.increment()
            print(progress_log)

            
            row.append(response_json['choices'][0]['message']['content'])
            return row

async def get_completion_list(content_list, max_parallel_calls, timeout):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(content_list))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
        content_list = content_list.values.tolist()
        content_labelled =  await asyncio.gather(*[
            get_completion(content, session, semaphore, progress_log) 
            for content in content_list
            ])
        return content_labelled



df = pd.read_csv(os.getenv('TWEET_CATEGORY_UNLABELLED_DATA_PATH'))
labelled_df = pd.DataFrame(columns=['Link','Date','Text','Category'])

content_list = df.sample(3000)

print(type(content_list))

loop = asyncio.get_event_loop()
completion_list = loop.run_until_complete(get_completion_list(content_list, 10, 30))


for completion in completion_list:
    completion[3] = completion[3].replace('Category: ', '')
    if completion[3] not in label_map:
        completion[3] = 7
    else:
        completion[3] = label_map[completion[3]]

    labelled_df.loc[len(labelled_df.index)] = completion

labelled_df.to_csv(os.getenv('TWEET_CATEGORY_LABELLED_DATA_PATH'), index=False)



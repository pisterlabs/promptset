import openai
import json
import re

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

def recommend_user_list(summary, dictionary_of_summaries):
    prompt = f'''You are an advanced recommendation engine specializing in analyzing patient summaries.
     Provided with a summary and a dictionary containing user names as keys and corresponding patient summaries as values,
     your objective is to assess and rank users based on the relevance of their summaries to the provided one.
     Your ultimate output should be a sorted python list of users, ordered by the relevance of their summaries to the given summary.
    <inp>
    Summary: {summary}
    Dictionary Of Summaries: {dictionary_of_summaries}
    </inp>

    OUTPUT FORMAT:
    Sorted List:
    '''

    pp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=14000,
        temperature=0.4
        )

    plan = pp['choices'][0]['message']['content']
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, plan)

    users = [user.strip() for user in matches[0].split(',')]
    users = [user.replace("'", "") for user in users]


    return users
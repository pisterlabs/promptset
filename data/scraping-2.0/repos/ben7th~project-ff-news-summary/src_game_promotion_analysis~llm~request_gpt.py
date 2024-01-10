from dotenv import load_dotenv
load_dotenv()

import openai
from retrying import retry

DEFAULT_MODEL_NAME = 'gpt-3.5-turbo'

@retry(stop_max_attempt_number=5, wait_fixed=2000) # 等待 2 秒后重试，最多重试 5 次
def request_gpt(prompt, model=DEFAULT_MODEL_NAME) -> str:
    result = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return result['choices'][0]['message']['content']

def request_gpt_with_system(prompt, system, model=DEFAULT_MODEL_NAME) -> str:
    result = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return result['choices'][0]['message']['content']
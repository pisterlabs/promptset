import openai
from dotenv import dotenv_values, load_dotenv
import time

config = dotenv_values("/workspace/Coding/lm-trainer/src/.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')

def gpt_call(
    model_name, # gpt-3.5-turbo, gpt-4
    prompt,
    max_retries=3,
    delay_between_retries=10
    ):
    
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Attempt {retries + 1} failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Waiting for {delay_between_retries} seconds before retrying...")
                time.sleep(delay_between_retries)

    return response
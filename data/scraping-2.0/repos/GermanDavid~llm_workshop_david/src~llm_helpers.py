import json
import requests
from openai import OpenAI
from decouple import config
openai_client = OpenAI(api_key=config("OPENAI_API_KEY"))
huggingfacehub_api_token = config("HUGGINGFACE_API_TOKEN")

def hf_ask(question: str, model_url="https://api-inference.huggingface.co/models/google/flan-t5-xxl") -> str:
    """Ask a question to Huggingface, apply it to every row of a pandas dataframe and return the answer"""
    def pandas_func(row) -> str:
        prompt = question.format(**(dict(row.items())))
        headers = {"Authorization": f"Bearer {huggingfacehub_api_token}"}
        response = requests.post(
            model_url, headers=headers, json={"inputs": prompt})
        if response.status_code != 200:
            return None
        return json.loads(response.content.decode("utf-8"))[0]['generated_text']
    return pandas_func


def chatgpt_ask(question: str, model_name="gpt-3.5-turbo") -> str:
    """Ask a yes/no question to chatgpt, apply it to every row of a pandas dataframe and return a boolean answer"""
    def pandas_func(row) -> bool:
        try:
            prompt = question.format(**(dict(row.items())))
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content.strip().lower()
            return answer == 'yes'  # Returns True if answer is 'yes', False otherwise
        except:
            return False  # or return None if you prefer to keep non-responses as None
    return pandas_func


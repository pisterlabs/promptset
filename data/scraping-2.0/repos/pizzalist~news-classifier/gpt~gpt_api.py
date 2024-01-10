import os

import openai
from dotenv import load_dotenv
import openai


load_dotenv()

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_version")

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def request_gpt3_api(
    sys_prompt: str,
    user_prompt: str,
    model: str = "teamlab-gpt-35-turbo",
    max_token: int = 2200,
    temperature: float = 0,
) -> str:
    response = openai.ChatCompletion.create(
        engine=model,
        messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        max_tokens=max_token,
        temperature=temperature,
    )
    return response.choices[0].message.content

def generate_news_summary(title,data):
    sys_template = read_prompt_template("sys_prompt.txt")
    user_template = read_prompt_template("user_prompt.txt")
    
    sys_prompt = sys_template.format(
        title=title,
        data=data,
    )
    print(sys_prompt, user_template)
    return request_gpt3_api(sys_prompt, user_template)

def generate_news_contents(data):
    sys_template = read_prompt_template("sys_prompt2.txt")
    user_template = read_prompt_template("user_prompt2.txt")

    sys_prompt = sys_template.format(
        data=data,
    )
    print(sys_prompt, user_template)
    return request_gpt3_api(sys_prompt, user_template)

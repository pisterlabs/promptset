import openai
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json
from typing import List

load_dotenv() # take environment variables from .env.

system_msg = """You are a LLM trainer. You help by responding with writing prompts that would generate the text input by the user.

You should use vague prompts so that the style is inherent and not just a question. Often times, writing is a metaphor or analogy, do not give literal prompts that about the metaphor or analogies themselves. Try to refrain from asking questions, but rather, give a prompt that would generate the text input by the user in a natural way. Lastly, please try to vary the prompts. Do not just ask questions or begin the prompt with "describe" or "explain".

Please generate 10 to 15 appropriate prompts. Your response should be limited to prompts only, separated by a new line. No bullet list, numbered list, or anything else."""


def generate_prompts(chunk : str) -> List[str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_msg,
                },
                {"role": "user", "content": f"{chunk}."},
            ],
            max_tokens=2000,
            temperature=0.5,
        )
        return response.choices[0]['message']['content'].strip().splitlines()
    except openai.error.APIError as e:
        print(f"openai error:  {e}")
        return None

def format_prompt(prompt : str, content : str) -> str:
    prompt_json     = json.dumps(prompt)
    completion_json = json.dumps(content)

    return f'{{"prompt": {prompt_json}, "completion": {completion_json}}}'
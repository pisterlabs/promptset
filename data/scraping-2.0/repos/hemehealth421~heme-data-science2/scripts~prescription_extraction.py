import json
import openai
import os
from scripts.output_format import *


from dotenv import load_dotenv
load_dotenv()

anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_model = os.environ.get("ANTHROPIC_MODEL")

openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_model = os.environ.get('OPENAI_MODEL')

openai.api_key = openai_api_key




def parse_json(json_str):
    text = ''.join(c for c in json_str if c > '\u001f')
    text = text.replace('\n', '\\n')
    try: 
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        print(f'Invalid JSON: {e}')
        return {}


def pre_extract(pre_text):
    gpt4_res = openai.ChatCompletion.create(model="gpt-4",
                                        messages=[{"role": "system", "content": pre_sys},
                                                  {"role": "user", "content": pre_text}],
                                        temperature=0)

    response_text = gpt4_res["choices"][0]["message"]["content"]
    parsed_res = parse_json(response_text)
    return parsed_res
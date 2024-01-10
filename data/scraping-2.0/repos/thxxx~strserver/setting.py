import openai
import deepl
import time
import json
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_BASE = os.getenv("API_BASE")
api_key = os.getenv("DEEPL_API_KEY")

translator = deepl.Translator(api_key)

def translate(text:str) -> str:
  result = translator.translate_text(text, source_lang="EN", target_lang="KO")
  return result.text

openai.api_type = "azure"
openai.api_key = AZURE_API_KEY
openai.api_base = API_BASE
openai.api_version = "2023-05-15"

def generate(prompt:str, systemMessage:str = "", model:str = "chatgpt", keys=[]) -> str:
    print("입력이 들어옴")

    response = openai.ChatCompletion.create(
        engine = model,
        messages=[
                {"role": "system", "content": f"You a helpful assistant, {systemMessage}"},
                {"role": "user", "content": prompt},
            ],
        )
    print("입력이 들어옴", response)
    
    if keys == []:
      res = response['choices'][0]['message']['content']
    else:
      res = format_check(response['choices'][0]['message']['content'], keys)

    response_body = {
        'data': res,
        "prompt_tokens": response['usage']['prompt_tokens'],
        "completion_tokens": response['usage']['completion_tokens'],
        "total_tokens": response['usage']['total_tokens']
    }

    return response_body


def modify_the_output(prompt:str) -> str:
  response = openai.ChatCompletion.create(
    engine = "chatgpt",
    messages=[
            {"role": "system", "content": "You a helpful format modifier"},
            {"role": "user", "content": prompt},
        ],
    )
  return response['choices'][0]['message']['content']


def format_check(output:str, keys:list = []):
  # Find the index of the first "{" and the last "}"
  first_curly_brace = output.find("{")
  last_curly_brace = output.rfind("}")

  if first_curly_brace != -1 and last_curly_brace != -1:
    processed = output[first_curly_brace:last_curly_brace+1]
  else:
    processed = output

  print(processed)
  try:
    output_json = json.loads(processed)
    print("안 맞는다고? ", set(list(output_json.keys()), set(keys)))
    assert set(keys) == set(list(output_json.keys()))
    return output_json
  except:
    # 포맷 고쳐줘 -> 출력
    st = time.time()
    print("포맷이 안맞아서 수정")

    format_prompt = f"""
      Modify below data to the python json format with keys : {keys}.

      {processed}
    """

    response = modify_the_output(format_prompt)

    first_curly_brace = response.find("{")
    last_curly_brace = response.rfind("}")

    if first_curly_brace != -1 and last_curly_brace != -1:
      processed_response = response[first_curly_brace:last_curly_brace+1]
    else:
      processed_response = response

    return json.loads(processed_response)
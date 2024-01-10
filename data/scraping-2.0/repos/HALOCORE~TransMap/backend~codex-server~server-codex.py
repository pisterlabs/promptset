from genericpath import isdir
import json
import os
import flask
from flask import request, jsonify, redirect, send_from_directory
from flask_cors import CORS
import openai
import time

print("=============== codex-server INIT ===============")

app = flask.Flask(__name__, static_url_path='/dashboard', static_folder='dashboard')
CORS(app)
app.config["DEBUG"] = True   # True

def read_json(fname):
  with open(fname, 'r') as f:
    return json.load(f)

def read_text(fname):
  with open(fname, 'r') as f:
    return f.read()

def save_to_path_autodir(filepath, content):
  dirpath = os.path.dirname(filepath)
  os.makedirs(dirpath, exist_ok=True)
  with open(filepath, 'w') as f:
    f.write(content)






################## APIs BEGIN ##################

@app.route("/")
def redirect_ui():
  return redirect("./help", code=302)

HELP_TEXT = """
This is codex-server (An proxy server for accessing Codex)

/help
    - GET -> help in plain text
/codextranslate
    - POST {source_language, target_language, source_code, max_length, temp} -> {target_code, error_msg}
/codextranslateanno/<preprocessor_name>
    - POST {anno_subsribtion, annotations, source_language, target_language, source_code} -> {target_code, error_msg}
"""

@app.route('/help')
def help_static():
  return app.response_class(response=HELP_TEXT, status=200, mimetype='text/plain')

@app.route('/codextranslate', methods=['POST'])
def codex_translate():
  if request.json is None:
    return jsonify({"result": None, "error_msg": "No posted data"})
  source_language = request.json["source_language"]
  assert source_language == "py"
  target_language = request.json["target_language"]
  assert target_language == "js"
  source_code = request.json["source_code"]
  max_length = request.json["max_length"]
  temp = request.json["temp"]
  is_beta = False 
  if "is_beta" in request.json: is_beta = request.json["is_beta"]
  engine = 'code-davinci-001'
  if "engine" in request.json: engine = request.json["engine"]
  try:
    target_code = translate_py_js(source_code, max_length, temp, is_beta, engine)
    result = {
      "target_code": target_code,
      "error_msg": None
    }
  except Exception as e:
    result = {
      "error_msg": str(e)
    }
  return jsonify(result)


@app.route('/codexpatch', methods=['POST'])
def codex_patch():
  if request.json is None:
    return jsonify({"result": None, "error_msg": "No posted data"})
  source_language = request.json["source_language"]
  assert source_language == "py"
  target_language = request.json["target_language"]
  assert target_language == "js"
  source_code = request.json["source_code"]
  target_code = request.json["target_code"]
  src_range = request.json["source_range"]
  tar_range = request.json["target_range"]
  insertion_prompt = request.json["insertion_prompt"]
  max_length = request.json["max_length"]
  temp = request.json["temp"]
  is_beta = False 
  if "is_beta" in request.json: is_beta = request.json["is_beta"]
  engine = 'code-davinci-001'
  if "engine" in request.json: engine = request.json["engine"]
  try:
    patched_code = translatepatch_py_js(source_code, target_code, src_range, tar_range, insertion_prompt, max_length, temp, is_beta, engine)
    result = {
      "patched_code": patched_code,
      "error_msg": None
    }
  except Exception as e:
    result = {
      "error_msg": str(e)
    }
  return jsonify(result)


@app.route('/codexpatchv2', methods=['POST'])
def codex_patch_v2():
  if request.json is None:
    return jsonify({"result": None, "error_msg": "No posted data"})
  source_language = request.json["source_language"]
  assert source_language == "py"
  target_language = request.json["target_language"]
  assert target_language == "js"
  source_code = request.json["source_code"]
  target_code_with_insert = request.json["target_code_with_insert"]
  src_range = request.json["source_range"]
  tar_range_str = request.json["target_range_str"]
  insertion_prompt = request.json["insertion_prompt"]
  max_length = request.json["max_length"]
  temp = request.json["temp"]
  is_beta = False 
  if "is_beta" in request.json: is_beta = request.json["is_beta"]
  engine = 'code-davinci-001'
  if "engine" in request.json: engine = request.json["engine"]
  try:
    patched_code = translatepatch_py_js_v2(source_code, target_code_with_insert, src_range, tar_range_str, insertion_prompt, max_length, temp, is_beta, engine)
    result = {
      "patched_code": patched_code,
      "error_msg": None
    }
  except Exception as e:
    result = {
      "error_msg": str(e)
    }
  return jsonify(result)


@app.route("/codexcomplete", methods=['POST'])
def codex_complete():
  if request.json is None:
    return jsonify({"result": None, "error_msg": "No posted data"})
  prompt = request.json["prompt"]
  max_length = request.json["max_length"]
  temp = request.json["temp"]
  stop_seqs = request.json["stop"]
  is_beta = False 
  if "is_beta" in request.json: is_beta = request.json["is_beta"]
  engine = 'code-davinci-001'
  if "engine" in request.json: engine = request.json["engine"]
  try:
    completion, timespan = complete_prompt(prompt, max_length, temp, is_beta, engine, stop_seqs)
    result = {
      "completion": json.loads(completion),
      "timespan": timespan,
      "error_msg": None
    }
  except Exception as e:
    result = {
      "error_msg": str(e)
    }
  return jsonify(result)

@app.route("/codexinsert", methods=['POST'])
def codex_insert():
  if request.json is None:
    return jsonify({"result": None, "error_msg": "No posted data"})
  prompt = request.json["prompt"]
  suffix = request.json["suffix"]
  max_length = request.json["max_length"]
  temp = request.json["temp"]
  stop_seqs = request.json["stop"]
  is_beta = False 
  if "is_beta" in request.json: is_beta = request.json["is_beta"]
  engine = 'code-davinci-001'
  if "engine" in request.json: engine = request.json["engine"]
  try:
    completion, timespan = complete_prompt_suffix(prompt, suffix, max_length, temp, is_beta, engine, stop_seqs)
    result = {
      "completion": json.loads(completion),
      "timespan": timespan,
      "error_msg": None
    }
  except Exception as e:
    result = {
      "error_msg": str(e)
    }
  return jsonify(result)

################## APIs END ##################







################## Codex BEGIN ##################

# https://beta.openai.com/docs/api-reference/completions/create

prompt1 = lambda pycode: f"""##### Translate this function from Python into Modern JavaScript
### Python

{pycode}
    
### JavaScript
"use strict";
"""

def patch_prompt1 (pycode, jscode, src_range, tar_range, insertion_prompt): 
  js_before = jscode[:tar_range[0]]
  js_after = jscode[tar_range[1]:]
  templ = f"""##### Translate this function from Python into Modern JavaScript
### Python

{pycode}
    
### JavaScript
"use strict";
{js_before}"""
  return templ, js_after

def patch_prompt_v2_1 (pycode, jscode_with_insert, src_range, tar_range_str, insertion_prompt):
  js_splits = jscode_with_insert.split("[insert]")
  assert len(js_splits) == 2
  js_before, js_after = js_splits
  templ = f"""##### Translate this function from Python into Modern JavaScript
### Python

{pycode}
    
### JavaScript
"use strict";
{js_before}"""
  return templ, js_after


PROMPT = prompt1
PATCH_PROMPT = patch_prompt1
PATCH_PROMPT_V2 = patch_prompt_v2_1
LOGPROBS = 1

import timeit
import sys
def complete_prompt(prompt, max_length, temp, is_beta=False, engine='code-davinci-001', stop=None):
  # pip install --upgrade openai
  if is_beta: print(f"[complete_prompt]  is_beta: {is_beta}  engine: {engine}", file=sys.stderr)
  openai.api_key = json.loads(read_text(("./drvtry-beta" if is_beta else "./drvtry")))
  while True:
    try:
      time1 = timeit.default_timer()
      if stop is None:
        res = openai.Completion.create(
          engine=engine,
          prompt=prompt,
          max_tokens=max_length,
          temperature=temp,
          logprobs=LOGPROBS,
          # echo=True,
          n=1
        )
      else:
        res = openai.Completion.create(
          engine=engine,
          prompt=prompt,
          max_tokens=max_length,
          temperature=temp,
          logprobs=LOGPROBS,
          stop=stop,
          # echo=True,
          n=1
        )
      time2 = timeit.default_timer()
      return str(res), time2 - time1
    except Exception as e:
      print(e)
      # time.sleep(30)
      # continue
      raise e


import requests
def complete_prompt_suffix(prompt, suffix, max_length, temp, is_beta=False, engine='code-davinci-001', stop=None):
  # pip install --upgrade openai
  if is_beta: print(f"[complete_prompt_suffix]  is_beta: {is_beta}  engine: {engine}", file=sys.stderr)
  api_key = json.loads(read_text(("./drvtry-beta" if is_beta else "./drvtry")))
  test_json_2 = {
    "model": engine,
    "prompt": prompt,
    "suffix": suffix,
    "max_tokens": max_length,
    "temperature": temp,
    "n": 1,
    "logprobs": LOGPROBS,
    "stream": False,
  }
  if stop is not None: test_json_2["stop"] = stop
  time1 = timeit.default_timer()
  resp = requests.post(
    "https://api.openai.com/v1/completions", 
    headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + api_key
    },
    json = test_json_2
  )
  time2 = timeit.default_timer()
  return resp.content, time2 - time1


def translate_py_js(pycode, max_length, temp, is_beta=False, engine='code-davinci-001'):
  prompt = PROMPT(pycode)
  save_to_path_autodir("./last-prompt.txt", prompt)
  resp, timespan = complete_prompt(prompt, max_length, temp, is_beta, engine)
  print("Trans Time: ", timespan)
  content_data = json.loads(resp)
  assert len(content_data["choices"]) == 1
  first_choice = content_data["choices"][0]
  finish_reason = first_choice["finish_reason"]
  text = first_choice["text"]
  if len(text.split("\n}")) < 2:
    return "//FAILED_TO_CLEAN\n" + text
  before, after = text.split("\n}")[:2]
  before = before + "\n}"
  return before

def translatepatch_py_js(pycode, jscode, src_range, tar_range, insertion_prompt, max_length, temp, is_beta=False, engine='code-davinci-002'):
  prompt, suffix = PATCH_PROMPT(pycode, jscode, src_range, tar_range, insertion_prompt)
  save_to_path_autodir("./last-insert-prompt.txt", prompt + "[INSERT]" + suffix)
  resp, timespan = complete_prompt_suffix(prompt, suffix, max_length, temp, is_beta, engine)
  print("Trans Time: ", timespan)
  content_data = json.loads(resp)
  if "choices" not in content_data or len(content_data["choices"]) != 1:
    print("======= Unexpected insertion resp =======\n", content_data)
    raise Exception("Unexpected content: " + str(resp))
  first_choice = content_data["choices"][0]
  finish_reason = first_choice["finish_reason"]
  text = first_choice["text"]
  # TODO: process
  return text

def translatepatch_py_js_v2(pycode, jscode_with_insert, src_range, tar_range_str, insertion_prompt, max_length, temp, is_beta=False, engine='code-davinci-002'):
  prompt, suffix = PATCH_PROMPT_V2(pycode, jscode_with_insert, src_range, tar_range_str, insertion_prompt)
  save_to_path_autodir("./last-insert-prompt.txt", prompt + "[INSERT]" + suffix)
  resp, timespan = complete_prompt_suffix(prompt, suffix, max_length, temp, is_beta, engine)
  print("Trans Time: ", timespan)
  content_data = json.loads(resp)
  if "choices" not in content_data or len(content_data["choices"]) != 1:
    print("======= Unexpected insertion resp =======\n", content_data)
    raise Exception("Unexpected content: " + str(resp))
  first_choice = content_data["choices"][0]
  finish_reason = first_choice["finish_reason"]
  text = first_choice["text"]
  # TODO: process
  return text

################## Codex END ##################


import argparse
parser = argparse.ArgumentParser(description='codex-server')
parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to use')
parser.add_argument('--port', type=int, default=8780, help='Port to use')
parser.add_argument('--reloader', type=bool, default=False, help='Using reloader')
args = parser.parse_args()

if __name__ == '__main__':
  app.run(host=args.host, port=int(args.port), use_reloader=args.reloader)
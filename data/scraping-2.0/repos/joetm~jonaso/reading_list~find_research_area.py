#!/usr/bin/python3

import os
import time
import openai
import json
from fnmatch import fnmatch
from tqdm import tqdm

# for killing and retrying hanging API calls
import signal
API_TIMEOUT = 10
# Define a custom exception for the timeout
class TimeoutException(Exception):
    pass
# Define a handler function that raises the TimeoutException
def timeout_handler(signum, frame):
    raise TimeoutException

from dotenv import dotenv_values
config = dotenv_values(".env")

# openai.organization = "YOUR_ORG_ID"
openai.api_key = config["OPENAI_KEY"]

temperature = 0 # default 1
top_p = 1  # default 1
n = 1 # How many chat completion choices to generate for each input message.
presence_penalty = 0 # Defaults to 0 # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
max_tokens = 5


PATTERN = "*.json"

input_folder = 'influencers'
output_folder = 'fields'

# system_prompt = """
# You are an autoregressive model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so.
# Since you are an autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. However: if a request begins with the string "vv" then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start, no summary at the end, and outputting only code for answers where code is appropriate.
# Your users are experts in AI, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, minimize vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.
# """

def call_api(prompt, model="gpt-3.5-turbo", timeout_duration=API_TIMEOUT):
  msgs = [{"role": "user", "content": prompt}]

  # Set the signal handler for the SIGALRM signal
  signal.signal(signal.SIGALRM, timeout_handler)
  # Schedule the alarm for timeout_duration seconds from now
  signal.alarm(timeout_duration)

  try:
    # print("Calling API...")
    signal.alarm(timeout_duration)
    response = openai.ChatCompletion.create(
      model=model,
      temperature=temperature,
      top_p=top_p,
      n=n,
      max_tokens=max_tokens,
      presence_penalty=presence_penalty,
      messages=msgs
    )
    # print("OK.")
    signal.alarm(0)
    return response
  except TimeoutException:
      print("***** Function call timed out, retrying...")
      # Cancel the alarm before retrying
      signal.alarm(0)
      return call_api(prompt, model=model)
  except openai.error.RateLimitError as e:
    retry_after = int(e.headers.get("retry-after", 10))
    print(f"Rate limit exceeded, waiting for {retry_after} seconds...")
    time.sleep(retry_after)
    return call_api(prompt, model=model)
  except Exception as e:
    print(f"API Error: {e}")
    time.sleep(10)  # To handle rate limits; adjust as n
    return call_api(prompt, model=model)

prompt_tpl = """Below are papers of an author and keywords. Your task is to provide the correct research area of this author, based on the papers and keywords. E.g., you will output HCI, crowdsourcing, or ML (for machine learning), NLP, etc. (there could be many other research areas).
Note that the keywords list represents a hierarchy of folders where publications are stored (but not in order). This means not all keywords accurately represent the research area.
You only ever respond with one single research area, without explanations. If there is more than one research area that could fit, you output the most likely one.
You prefer short research areas over longer ones. For example, you output 'social media' instead of 'social media analytics', 'NLP' instead of 'Natural Language Processing (NLP), and 'ML' instead of 'Machine Learning').

Keywords:
<keywords>

Papers:
"""


for path, subdirs, files in os.walk(input_folder, followlinks=False):

  # pbar = tqdm(total=len(files))

  for name in files:

    fullpath = os.path.join(path, name)

    if not fnmatch(name, PATTERN):
      # pbar.update(1)
      continue

    outfile = os.path.join(output_folder, name.replace(".json", ".txt"))

    # cache check
    if os.path.exists(outfile):
      # pbar.update(1)
      continue

    with open(fullpath, 'rb') as f:
      doc = json.load(f)

    keywords = ", ".join(doc['keywords'])
    print("Keywords:", keywords)

    pubs = [ d['title'] for d in doc['docs'] ]
    pubs = "- " + "\n- ".join(pubs)
    print(pubs)

    prompt = prompt_tpl + pubs
    prompt = prompt.replace("<keywords>", keywords)
    # print(prompt)
    # import sys; sys.exit()

    res = call_api(prompt)
    field = res['choices'][0]['message']['content']
    field = field.strip("\"").lower()
    field = field.replace("research area:", "")
    field = field.strip(',')
    field = field.strip()

    print(name.replace(".json", ":"), field)

    with open(outfile, 'w') as f:
      f.write(field)

    print('===')
    # pbar.update(1)


# manually fix some detected errors
error_fixes = [
  ("natural language processing (nlp)", "nlp"),
  ("machine learning (ml)", "ml/ai"),
  ("artificial intelligence (ai)", "ml/ai"),
  ("ubiquitous computing", "ubicomp"),
  ("machine learning", "ml/ai"),
  ("multimodal retrieval", "ml/ai"),
  ("law or legal", "legal"),
  ("no research area.", "none"),
  ("research area: hci", "hci"),
  ("privacy-preserving technologies", "privacy"),
  # merge ml and ai
  ("ai", "ml/ai"),
  ("ml", "ml/ai"),
  ("multimodal models", "ml/ai"),
  # merge linked data and semantic web
  # ("linked data", "linked data/semantic web"),
  # ("semantic web", "linked data/semantic web"),
]
for path, subdirs, files in os.walk(output_folder, followlinks=False):
  pbar = tqdm(total=len(files))
  for name in files:
    fullpath = os.path.join(path, name)
    with open(fullpath, 'r') as f:
      doc = f.readline()
    # if there are multiple research areas, take only the last one
    doc = doc.strip(',')
    if ',' in doc:
      print('fix:', name, doc)
      doc = doc.split(",")[-1]
    # check the hardcoded error fixes
    for e in error_fixes:
      if doc == e[0]:
        print('fix:', name, doc)
        # import sys; sys.exit()
        with open(fullpath, 'w') as out:
          out.write(e[1])


# dump the research areas into one file
data = {}
for path, subdirs, files in os.walk(output_folder, followlinks=False):
  for name in tqdm(files):
    fullpath = os.path.join(path, name)
    with open(fullpath, 'rb') as f:
      field = f.readline().decode('utf-8')
      field = field.lower().strip()
    id = name.replace('.txt', '')
    data[id] = field
with open('fields.json', 'w') as f:
  json.dump(data, f)

# what are the top fields?
from collections import defaultdict
from itertools import islice
freq = defaultdict(int)
for key in data:
  freq[data[key]] += 1
freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
with open('fields-freq.json', 'w') as f:
  json.dump(freq, f)
capped_freq = dict(islice(freq.items(), 20))
for area in capped_freq:
  capped_freq[area] = {'n': capped_freq[area]}
with open('fields-top20.json', 'w') as f:
  json.dump(capped_freq, f)

# instead of json, writes this as a .css style file
tpl = """
.ra.%s {
  background-color: %s;
}
"""
css_string = ''
i = 0
colors = ['#FFDAB9', '#FFB6C1', '#B0E0E6', '#AFEEEE', '#FFFFE0', '#E0FFFF', '#F0E68C', '#E6E6FA', '#FFF0F5', '#FFE4E1', '#F5DEB3', '#F0FFF0', '#FAFAD2', '#D3FFCE', '#F4C2C2', '#89CFF0', '#FAF0E6', '#FDFD96', '#ECCAFF', '#C1F0F6',]
for area in capped_freq:
  css_string += tpl % (area.replace(" ", "_"), colors[i])
  i += 1
with open('./fields.css', 'w') as f:
  f.write(css_string)

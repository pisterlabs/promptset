#!/usr/bin/env python3

import json
import re
import requests
import os
from multiprocessing.pool import ThreadPool

import openai
import srt

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from requests.exceptions import HTTPError
from sklearn.cluster import KMeans


VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
VICUNA = "http://fc6000.sf.archive.org:8000/v1"
OUTPUT_FOLDER_NAME = "summaries"

LLM_MODELS = {
  "OpenAI": "gpt-3.5-turbo",
  "Vicuna": "text-embedding-ada-002",
}

IDDTRE = re.compile(r"^.+_(\d{8}_\d{6})")

CHANNELS = [
  "ESPRESO",
  "RUSSIA1",
  "RUSSIA24",
  "1TV",
  "NTV",
  "BELARUSTV",
  "IRINN",
]

LM = "OpenAI" # language model
CK = 30 # chunk size
CT = 20 # cluster count
DT = (datetime.now() - timedelta(hours=30)).date().strftime("%Y%m%d") # date
LG = "English" # language

THREAD_COUNT = 10

def load_srt(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.srt")
  r.raise_for_status()
  return r.content


def load_inventory(ch, dt, lg):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  return pd.json_normalize(r.json(), record_path="shows").sort_values("start_time", ignore_index=True)


def create_doc(txt, id, start, end):
  return Document(page_content=txt, metadata={"id": id, "start": round(start.total_seconds()), "end": round(end.total_seconds())})


def chunk_srt(sr, id, lim=3.0):
  docs = []
  ln = 0
  txt = ""
  start = end = timedelta()
  for s in srt.parse(sr.decode()):
    cl = (s.end - s.start).total_seconds()
    if ln + cl > lim:
      if txt:
        docs.append(create_doc(txt, id, start, end))
      ln = cl
      txt = s.content
      start = s.start
      end = s.end
    else:
      ln += cl
      txt += " " + s.content
      end = s.end
  if txt:
    docs.append(create_doc(txt, id, start, end))
  return docs


def load_chunks(inventory, lg, ck):
  chks = []
  for i, r in inventory.iterrows():
    try:
      sr = load_srt(r.id, lg)
    except HTTPError as _:
      continue
    chks += chunk_srt(sr, r.id, lim=ck)
  return chks


def load_vectors(d, llm):
  embed = OpenAIEmbeddings(model=LLM_MODELS[llm])
  result = embed.embed_query(d.page_content)
  return result

def select_docs(dt, ch, lg, lm, ck, ct):
  print("loading chunks...")
  docs = load_chunks(inventory, lg, ck)
  docs_list = [(d,lm) for d in docs]

  print("loading vectors...")
  with ThreadPool(THREAD_COUNT) as pool:
    vectors = pool.starmap(load_vectors, docs_list)

  print("number of vectors =", len(vectors))
  kmeans = KMeans(n_clusters=ct, random_state=10, n_init=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]


def id_to_time(id, start=0):
  dt = IDDTRE.match(id).groups()[0]
  return datetime.strptime(dt, "%Y%m%d_%H%M%S") + timedelta(seconds=start)


def get_summary(d, llm):
  msg = f"""
  ```{d}```

  Create the most prominent headline from the text enclosed in three backticks (```) above, describe it in a paragraph, assign a category to it, determine whether it is of international interest, determine whether it is an advertisement, and assign the top three keywords in the following JSON format:

  {{
    "title": "<TITLE>",
    "description": "<DESCRIPTION>",
    "category": "<CATEGORY>",
    "international_interest": true|false,
    "advertisement": true|false,
    "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"]
  }}
  """
  res = openai.ChatCompletion.create(
    model=LLM_MODELS[llm],
    messages=[{"role": "user", "content": msg}]
  )
  result = json.loads(res.choices[0].message.content.strip())
  result = result | d.metadata
  result["transcript"] = d.page_content.strip()
  return result


if LM == "Vicuna":
  openai.api_key = "EMPTY"
  openai.api_base = VICUNA

# attempt to create ./OUTPUT_FOLDER_NAME
try:
  os.mkdir(f"./{OUTPUT_FOLDER_NAME}")
except FileExistsError:
  pass

for ch in CHANNELS:
  print(f"---\nstarting {ch}...")
  try:
    print("loading inventory...")
    inventory = load_inventory(ch, DT, LG)
  except HTTPError as _:
    print(f"Inventory for `{ch}` channel is not available for `{DT[:4]}-{DT[4:6]}-{DT[6:8]}` yet, try selecting another date!", icon="⚠️")

  print("loading documents...")
  seldocs = select_docs(DT, ch, LG, LM, CK, CT)

  print("begin summarizing each document...")

  summary_args = [(d,LM) for d in seldocs]
  with ThreadPool(THREAD_COUNT) as pool:
    summaries = pool.starmap(get_summary, summary_args)

  print("writing results...")
  with open(f"{OUTPUT_FOLDER_NAME}/{ch}-{DT}-{LM}-{LG}.json", 'w+') as f:
    f.write(json.dumps(summaries, indent=2))
  print(f"finished {ch}")

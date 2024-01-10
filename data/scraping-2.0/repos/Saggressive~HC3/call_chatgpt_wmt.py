# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import datasets
from datasets import load_dataset,load_from_disk
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
openai.api_key="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"

def zh2en(row):
  # print(row)
  row=row["translation"]
  en,zh=row["en"],row["zh"]
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "You are a helpful assistant that translates Chinese to English."},
      {"role": "user", "content": f'Translate the following Chinese text to English.: {zh}'},
    ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"
  data = {"source":zh,"gpt_gen":gpt_gen,"target":en}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def en2zh(row):
  row=row["translation"]
  en,zh=row["en"],row["zh"]
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "你是个将英文翻译成中文的助手"},
      {"role": "user", "content": f'将下列英文文本翻译成中文。: {en}'},
    ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"

  data = {"source":en,"gpt_gen":gpt_gen,"target":zh}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def translate_zh_en(data,subset,mode,sample):
  # dataset = load_dataset(data,subset,streaming=True)
  dataset = load_from_disk(f"{data}_{subset}")
  dataset = dataset[mode]
  
  dataset = dataset.shuffle(seed=42)
  data_mode = []
  for index,i in enumerate(dataset):
    if index<sample:
      data_mode.append(i)
    else:
      break
    
  folder = "en_zh"
  os.makedirs(folder,exist_ok=True)
  with Pool(processes=64) as p:
    zh_to_en = p.map(zh2en, tqdm(data_mode))
    file_path = folder + os.sep + f"zh2en_{mode}.jsonl"
    with open(file_path,"w") as f:
      f.writelines(zh_to_en)

  with Pool(processes=64) as p:
    zh_to_en = p.map(en2zh, tqdm(data_mode))
    file_path = folder + os.sep + f"en2zh_{mode}.jsonl"
    with open(file_path,"w") as f:
      f.writelines(zh_to_en)

def fr2en(row):
  row=row["translation"]
  en,fr=row["en"],row["fr"]
  try:
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "You are a helpful assistant that translates French to English."},
    {"role": "user", "content": f'Translate the following French text to English.: {fr}'},
  ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"
  data = {"source":fr,"gpt_gen":gpt_gen,"target":en}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def translate_fr_en(data,subset,mode,sample):
  folder = "en_fr"
  os.makedirs(folder,exist_ok=True)
  # dataset = load_dataset(data,subset,streaming=True)
  dataset = load_from_disk(f"{data}_{subset}")
  dataset = dataset[mode]
  
  dataset = dataset.shuffle(seed=42)
  data_mode = []
  for index,i in enumerate(dataset):
    if index<sample:
      data_mode.append(i)
    else:
      break
    
  with Pool(processes=64) as p:
    fr_to_en = p.map(fr2en, tqdm(data_mode))
    file_path = folder + os.sep + f"fr2en_{mode}.jsonl"
    with open(file_path,"w") as f:
      f.writelines(fr_to_en)

def de2en(row):
  row=row["translation"]
  en,de=row["en"],row["de"]
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "You are a helpful assistant that translates German to English."},
      {"role": "user", "content": f'Translate the following German text to English.: {de}'},
    ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"
  data = {"source":de,"gpt_gen":gpt_gen,"target":en}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def translate_de_en(data,subset,mode,sample):
  folder = "en_de"
  os.makedirs(folder,exist_ok=True)
  # dataset = load_dataset(data,subset,streaming=True)
  dataset = load_from_disk(f"{data}_{subset}")
  dataset = dataset[mode]
  
  dataset = dataset.shuffle(seed=42)
  data_mode = []
  for index,i in enumerate(dataset):
    if index<sample:
      data_mode.append(i)
    else:
      break
    
  with Pool(processes=64) as p:
    de_to_en = p.map(de2en, tqdm(data_mode))
    file_path = folder + os.sep + f"de2en_{mode}.jsonl"
    with open(file_path,"w") as f:
      f.writelines(de_to_en)

def ro2en(row):
  row=row["translation"]
  en,ro=row["en"],row["ro"]
  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "You are a helpful assistant that translates Romanian to English."},
      {"role": "user", "content": f'Translate the following Romanian text to English.: {ro}'},
    ])
    gpt_gen=response["choices"][0]["message"]["content"]
  except:
    gpt_gen="null"
  data = {"source":ro,"gpt_gen":gpt_gen,"target":en}
  data = json.dumps(data,ensure_ascii=False)
  data = data+"\n"
  return data

def translate_ro_en(data,subset,mode,sample):
  folder = "en_ro"
  os.makedirs(folder,exist_ok=True)
  # dataset = load_dataset(data,subset,streaming=True)
  dataset = load_from_disk(f"{data}_{subset}")
  dataset = dataset[mode]
  
  dataset = dataset.shuffle(seed=42)
  data_mode = []
  for index,i in enumerate(dataset):
    if index<sample:
      data_mode.append(i)
    else:
      break
  
  with Pool(processes=64) as p:
    ro_to_en = p.map(ro2en, tqdm(data_mode))
    file_path = folder + os.sep + f"ro2en_{mode}.jsonl"
    with open(file_path,"w") as f:
      f.writelines(ro_to_en)

if __name__=="__main__":
  sample = 6200
  data,subset= "wmt19","zh-en"
  translate_zh_en(data,subset,mode="train",sample=sample)

  data,subset= "wmt14","fr-en"
  translate_fr_en(data,subset,mode="train",sample=sample)

  data,subset= "wmt16","de-en"
  translate_de_en(data,subset,mode="train",sample=sample)

  data,subset= "wmt16","ro-en"
  translate_ro_en(data,subset,mode="train",sample=sample)

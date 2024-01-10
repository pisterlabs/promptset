import openai
import pandas as pd
from tqdm import tqdm
import glob
import json
import time

openai.api_key = ""
openai.api_base =  ""
openai.api_type = ''
openai.api_version = ''

id_to_json = {}

for filename in glob.glob("../data/maindata/tables/json/*"):
  f = open(filename,'r')
  data = json.load(f)
  id_to_json[int(filename.split("/")[-1].split(".")[0])] = data

def chatgpt_result(question):
    completion=openai.ChatCompletion.create(
      engine="gpt4_deployment", # this is "ChatGPT" $0.002 per 1k tokens
      messages=[{"role": "user", "content": question}]
    )
    print(completion.usage)
    return completion.choices[0].message.content

def prepare_input(json_table):
  return f""" Convert the following json into a knowledge graph with subject, predicate and object json  {json_table}"""


df_kg_original = pd.read_csv("kg.csv")

df_kg = pd.DataFrame()
start = len(df_kg_original)
count = 0
jsons = []
kgs = []

for t in tqdm(range(start, len(id_to_json))):
  count += 1
  jsons.append(id_to_json[t])

  input_text = prepare_input(id_to_json[t])

  j = 0
  while True:
    try:
        if j == 3:
          result = "haa"
        else:
          result = chatgpt_result(input_text)
        time.sleep(60)
        break
    except:
        j += 1
        print("....halt....")
        time.sleep(20)

  kgs.append(result)
  print(result)

  if count%5 == 0:
    df_kg['json'] = jsons
    df_kg['kg'] = kgs
    df_kg_original = pd.concat([df_kg_original, df_kg], ignore_index=True, sort=False)
    df_kg_original.to_csv('kg.csv',index=False)
    df_kg_original=pd.read_csv('kg.csv')

    jsons = []
    kgs = []
    df_kg= pd.DataFrame()

df_kg['json'] = jsons
df_kg['kg'] = kgs
df_kg_original = pd.concat([df_kg_original, df_kg], ignore_index=True, sort=False)
df_kg_original.to_csv('kg.csv',index=False)
df_kg_original=pd.read_csv('kg.csv')

####-------####
#knowledge graph preprocessing 
df = pd.read_csv("kg.csv")
json_to_kg = {}
new_json_to_kg = {}
for json, kg in zip(df["json"],df["kg"]):
  json_to_kg[json] = kg
  new_json_to_kg[json] = kg

ids1 = []
for id in id_to_json:
  try:
    kg = ast.literal_eval(json_to_kg[str(id_to_json[id])])
  except:
    ids1.append(id)

ids2 = []
for id in id_to_json:

  if id in ids1:
    continue

  try:
    kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])
    object1 = kg[0]["object"]
  except:
    try:
      kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])
      object1 = kg[0]["Object"]
    except:
      ids2.append(id)

ids3 = []
for id in ids2:
  try:
    kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])['knowledge_graph']
    new_json_to_kg[str(id_to_json[id])] = str(kg)
  except:
    try:
      kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])['edges']
      new_json_to_kg[str(id_to_json[id])] = str(kg)
    except:
      try:
        kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])['graph']
        new_json_to_kg[str(id_to_json[id])] = str(kg)
      except:
        try:
          kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])['graphs']
          new_json_to_kg[str(id_to_json[id])] = str(kg)
        except:
          try:
            kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])['KnowledgeGraph']
            new_json_to_kg[str(id_to_json[id])] = str(kg)
          except:
            try:
              kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])["knowledgeGraph"]["nodes"]
              new_json_to_kg[str(id_to_json[id])] = str(kg)
            except:
              try:
                kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])["knowledgeGraph"]
                new_json_to_kg[str(id_to_json[id])] = str(kg)
              except:
                try:
                  kg = ast.literal_eval(new_json_to_kg[str(id_to_json[id])])["triples"]
                  new_json_to_kg[str(id_to_json[id])] = str(kg)
                except:
                  ids3.append(id)
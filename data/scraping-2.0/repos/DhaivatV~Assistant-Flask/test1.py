import requests
import json
import pandas as pd
import os
import openai
import requests
from dotenv import load_dotenv
import nltk

load_dotenv()

openai.api_key =  os.getenv('API_KEY')

with open('data.json', 'r') as file:
    data = json.load(file)


df = pd.DataFrame(data['details']) 
d_c_list = df['name'].to_list()
domain_list = d_c_list[:10]
committee_list = d_c_list[10:23]
for i in range(len(df)):
    name = df.loc[i, 'name']
    if name in domain_list:
        df.loc[i, 'domain/committee'] = "domain"
    elif name in committee_list :
        df.loc[i, 'domain/committee'] = "committee"
    else:
        df.loc[i, 'domain/committee'] = "team"

df['combined_info'] = df['name'] + " " + df['domain/committee'] + " " + df['organizers'] + " " + df['description'] 
list_info = df['combined_info'].to_list()

def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def fetch_row_by_info(info):
    for index, row in df.iterrows():
        if info in row['combined_info']:
            res = row.to_list()
            res_dict = row.to_dict()
            return [res, res_dict]
    return None

def get_row_content_similarity(row_content, query):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    api_token = "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"
    headers = {"Authorization": f"Bearer {api_token}"}

    def req(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.json()

    data = req(
      {
          "inputs": {
              "source_sentence": query.lower(),
              "sentences": row_content
          }
      }


        )
    
    smi_index = []

    for similarity in data:
    
        if similarity>0.9:
            smi_index.append(similarity)
        
        
        elif similarity>0.7:
            smi_index.append(similarity)

        elif similarity>0.5:
            smi_index.append(similarity)

        elif similarity>0.4:
            smi_index.append(similarity)


    if len(smi_index) ==1:
        info_index1 = data.index(smi_index[0])

    elif len(smi_index) >1:
        smi_index.sort() 
        info_index1 = data.index(smi_index[-1])
        info_index2 = data.index(smi_index[-2])
        info_index3 = data.index(smi_index[-3])

    context = [row_content[info_index1] , row_content[info_index2], row_content[info_index3]]
    return (context)
  


def similar_context(query):

  API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
  api_token = "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"
  headers = {"Authorization": f"Bearer {api_token}"}

  def req(payload):
      response = requests.post(API_URL, headers=headers, json=payload)
      return response.json()

  data = req(
      {
          "inputs": {
              "source_sentence": query.lower(),
              "sentences": list_info
          }
      }


  )
              
  smi_index = []

  for similarity in data:
    
    if similarity>0.9:
       smi_index.append(similarity)
    
    
    elif similarity>0.7:
       smi_index.append(similarity)

    elif similarity>0.5:
       smi_index.append(similarity)

    elif similarity>0.4:
       smi_index.append(similarity)


  if len(smi_index) ==1:
     exact_info_index = data.index(smi_index[0])

  elif len(smi_index) >1:
     smi_index.sort() 
     exact_info_index = data.index(smi_index[-1])

  else:
      return("Sorry!! Not enough info provided in question")
  
  row_content = fetch_row_by_info(list_info[exact_info_index])

  res1 = json.dumps(row_content[1])

  x= count_tokens(res1)
  print(x)
  if (x>2000):
    context = get_row_content_similarity(row_content[0],query)
    keys = []
    def get_key(val):
    
        for key, value in row_content[1].items():
            if val == value:
                return key
    
        return "key doesn't exist"
    
    result = {}

    for items in context:
        con_key = get_key(items)
        keys.append(con_key)
    


    for key in keys:

        if key in row_content[1].keys():
            result[key]=row_content[1].get(key)
    
    return (json.dumps(result))
  

  
  return json.dumps(res1)

def get_response(prompt_input):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt_input,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
    )

    return (((response.get('choices'))[0]).get('text'))



def query_response(user_query) :
  
    context = (similar_context(user_query))+"\n\n"
    print(context)
    if context == "Sorry!! Not enough info provided in question":
        return (context)
    else:
        final_query = "Give precise answer for the question asked from the data below\n\n"+ context + f'\n\nQ:{user_query}\n\nA:'
        res = (get_response(final_query))
        return res
    
          

def query(user_query):
    response = query_response(user_query)
    print(response)


query("who is convenor of aaruush")
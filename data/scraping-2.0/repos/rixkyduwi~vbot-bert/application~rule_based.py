import torch,time,numpy as np
from flask import jsonify
from transformers import BertTokenizerFast, BertForQuestionAnswering
from datetime import datetime
from . import mysql
import openai
import textwrap,re
import urllib, json,os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("API_CHATGPT")
#print(key)
openai.api_key = key
device = "cuda" if torch.cuda.is_available() else "cpu" 
torch.device(device) 
modelCheckpoint = "indolem/indobert-base-uncased"

model_directory = os.path.abspath(os.path.join(__file__, "../../bert_model/model.bin")) 
model = torch.load(model_directory,map_location=torch.device('cpu'))
model.to(device)
tokenizer = BertTokenizerFast.from_pretrained(modelCheckpoint)
start_time = time.time()

gmaps_rumah_sakit = "Rumah Sakit terdekat => "
gmaps_puskesmas = "Puskesmas terdekat => "
gmaps_apotek = "Apotek terdekat => "
gmaps_urut = "Tempat pijat / urut disekitar anda => "

link_rumah_sakit = "https://www.google.com/maps/search/Rumah_sakit/"
link_puskesmas = "https://www.google.com/maps/search/Puskesmas/"
link_apotek = "https://www.google.com/maps/search/Apotek/"
link_urut = "https://www.google.com/maps/search/Urut/"


def convertTuple1(tup):
    return ''.join([str(x) for x in tup])

# dari database

def rules_prediction(rules,question):
  respon_model = []
  dictlogs = {}
  penyakit_terdeteksi = []
  print(rules)
  print(question)
  list_gejala = re.sub("[,]", " ", question).split()
  gejala = []
  for string in list_gejala:
      if "dan" in string:
        continue
      new_string = string.strip()
      new_string = new_string.lower()
      gejala.append(new_string)
  print(gejala)
  matched_diseases = set()
  detailed_matched_diseases = set()
  for symptoms, disease in rules.items():
        matched_symptoms = sum(symptom in gejala for symptom in symptoms)
        print(matched_symptoms)
        if matched_symptoms == len(gejala):
            matched_diseases.add(disease)
        elif matched_symptoms == 1:
          if "adalah" in disease:
            matched_diseases.add(disease)
          elif "saya" in gejala:
            matched_diseases.add(disease)
    
  list(matched_diseases) if matched_diseases else ['Tidak Diketahui']
  hasil_deteksi = list(matched_diseases)
  if hasil_deteksi ==[]:
    dictlogs.update({"status": False,"deskripsi": "Maaf saya tidak tahu penyakit anda"})
  else:
    jawaban = ', '.join(hasil_deteksi)
    if "adalah" in jawaban:
      dictlogs.update({"status": True,"jawaban": ', '.join(hasil_deteksi)})
    else:
      dictlogs.update({"status": True,"jawaban": "Beberapa penyakit yang saya temukan berdasarkan gejala anda yaitu: "+', '.join(hasil_deteksi)})
   
  respon_model.append(dictlogs)    
  
  return jsonify(respon_model)

def random_question(ask):
  prompt = (ask)

  completions = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5,
  )

  message = completions.choices[0].text
  print(message)
  
  return message
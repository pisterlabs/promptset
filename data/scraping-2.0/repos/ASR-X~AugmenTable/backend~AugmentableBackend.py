from configs import *
import os
import openai
import requests


import json
dangers=json.loads(dangers)


def getIngs(barcode):
  data=requests.get("https://world.openfoodfacts.org/api/v0/product/{}.json".format(barcode))
  return [x[3:] for x in data.json()["product"]["ingredients_original_tags"]]

def getPBads(conditions):#potential bad ingredients
  global dangers
  bads=[]
  for c in conditions:
    bads+=dangers[c]
  return list(set(bads))


def getBads(conditions,barcode):
  pbads=getPBads(conditions)
  ings=getIngs(int(barcode))
  return list(set(pbads) & set(ings))

def getFeedback(conditions,ingredient):
  global gptprompt
  condStr=""
  if len(conditions)==0:
    condStr="no condition"    
  else:
    condStr=conditions[0]
    conditions.pop(0)
    while len(conditions):
      if len(conditions)==1:
        condStr+=((", and " if ("," in condStr) else " and ")+conditions[0])
      else:
        condStr+=(", "+conditions[0])
      conditions.pop(0)
  response = openai.Completion.create(engine="curie", prompt=gptprompt+conditionsPrompt.format(condStr,ingredient),
                                        temperature=0.4,max_tokens=200,top_p=1,frequency_penalty=0.13,presence_penalty=0.08,stop=["\n"]
  )
  res=response['choices'][0]['text']
  return res

def isGood(conditions,ingredient):
  global gptprompt
  ret=""
  while not ("yes" in ret[-5:] or "no" in ret[-5:]):
    condStr=""
    if len(conditions)==0:
      condStr="no condition"    
    else:
      condStr=conditions[0]
      conditions.pop(0)
      while len(conditions):
        if len(conditions)==1:
          condStr+=((", and " if ("," in condStr) else " and ")+conditions[0])
        else:
          condStr+=(", "+conditions[0])
        conditions.pop(0)
    response = openai.Completion.create(engine="curie-instruct-beta", prompt=gptprompt1+conditionsPrompt.format(condStr,ingredient),
                                        temperature=0.4,max_tokens=200,top_p=1,frequency_penalty=0.13,presence_penalty=0.08,stop=["condition:"]
    )
    ret=response['choices'][0]['text']
    print(ret)
  return "yes" in ret[-3:]


from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from time import sleep
import re
from typing import List

class BarcodeIn(BaseModel):
    barcode : str
class IngredientIn(BaseModel):
    ingredient : str
class ConditionsIn(BaseModel):
    conditions: List[str] = []
class BadsOut(BaseModel):
    bads: List[str] = []
class ImageIn(BaseModel):
    base64: str

class ImageOut(BaseModel):
    html: str

class TextOut(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods='*',
    allow_headers='*',
)

#https://www.nutrition.gov/topics/diet-and-health-conditions

@app.api_route("/", methods=["GET", "POST"])
async def root():
    return 'alive'

@app.post('/push')
async def push(request: Request, background_tasks: BackgroundTasks):
  reqjson = await request.json()
  background_tasks.add_task(simplify, reqjson)
  return Response(status_code=200)
  

@app.post('/getBads', response_model=BadsOut)
async def ask(conditions: ConditionsIn,barcodein: BarcodeIn):
  conditions=json.loads(conditions.json())
  barcodein=json.loads(barcodein.json())

  print(conditions)
  return {"bads":getBads([x for x in conditions["conditions"]],barcodein['barcode'])}

@app.post('/getFeedback', response_model=TextOut)
def ask(conditions: ConditionsIn, ingredientin: IngredientIn):
  conditions=json.loads(conditions.json())
  ingredientin=json.loads(ingredientin.json())
  return {"text":getFeedback([x for x in conditions["conditions"]],ingredientin["ingredient"])}

import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(80, bind_tls=True)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=80)

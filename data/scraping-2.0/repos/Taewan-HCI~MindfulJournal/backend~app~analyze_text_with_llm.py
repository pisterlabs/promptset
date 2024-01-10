# -*- coding: utf-8 -*-
import base64
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from konlpy.tag import Okt
import json
import openai
from dotenv import load_dotenv
import os
import time
from kiwipiepy import Kiwi
from collections import Counter
import firebase_admin
import prompts
from firebase_admin import credentials
from firebase_admin import firestore

kiwi = Kiwi()
app = FastAPI()
origins = ["*", "http://localhost:3000", "localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 구글 Firebase인증 및 openAI api key 설정
load_dotenv()
gptapi = os.getenv("gptkey")
My_OpenAI_key = gptapi
openai.api_key = My_OpenAI_key

firebase_key = os.getenv("dbkey")

cred = credentials.Certificate(firebase_key)
app_1 = firebase_admin.initialize_app(cred)
db = firestore.client()

#firebase에 user ID와 doc num에 해당하는 frequency, summary 데이터를 저장하는 함수
def uploadFirebase(user, num, frequency, summary):
    doc_ref = db.collection(u'session').document(user).collection(u'diary').document(num)
    doc_ref.set({
        u'wordFrequency': frequency,
        u'eventSummary': summary
    }, merge=True)
    
#firebase에 user의 num에 해당하는 diary에서 user가 말한 부분만 string format로 리턴하는 함수
def get_monologue(user, num):
    diary_ref = db.collection(u'session').document(user).collection(u'diary').document(num)
    diary = diary_ref.get().to_dict()
    conversation = diary.get('conversation')[1::2]
    monologue = ""
    for c in conversation:
        monologue += c['content']
    return monologue

#GPT에 text를 넣으면 나온 단어의 빈도수를 분석한 결과를 {word, frequency, sentiment}의 JSON Array로 리턴하는 함수
def get_word_frequency(text):
    messages = [{"role": "system",
                 "content": prompts.nlp_eng_prompt},
                {"role": "user",
                 "content": "monologue: ''' " + text + " '''  \n result:"}]
    
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stop=['User: '],
        max_tokens=2048,
        temperature=0
    )
    answer = completion
    result = answer["choices"][0]["message"]['content']

    json_data = json.loads(result.replace("\'", "\""))
    return_json = []
        
    주요품사 = ['NNG', 'NNP', 'VV', 'VA', 'XR', 'SL']
        
    #kiwi를 사용해서 한번 더 후처리
    for j in json_data : 
        t = kiwi.tokenize(j['word'])
        if t[0].tag in 주요품사:
            return_json.append(j)
    return return_json 
    
#text를 넣으면 event/ emotion&thought를 분석한 다음 {event: event, emotion: emotion} 형식으로 리턴하는 함수
def get_event_summary(text):
    messages = [{"role": "system",
                 "content": prompts.summary_prompt
                },
                {"role": "user",
                 "content": "here is a diary" + text}]
        
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stop=['User: '],
        max_tokens=2048,)
    
    answer = completion
    result = answer["choices"][0]["message"]['content']
    

    event = result.split('Event:')[1].split('Emotion/Thought:')[0]
    emotion = result.split('Event:')[1].split('Emotion/Thought:')[1]
    event_summary = {
        "event": event,
        "emotion": emotion
    }
    return event_summary

#테스트용 api
@app.get('/')
def hello_world():
    return {'message': 'Hello, World!'}

@app.post("/upload")
async def uploadDB(request: Request):
    body = await request.json()
    user = body['user']
    num = body['num']
    
    start_time = time.time()
    
    monologue = get_monologue(user, num)
    
    for attempt in range(3):
        try:
            frequency = get_word_frequency(monologue)
        except:
            frequency = "Word Frequency Format Save Error Occured"
        else:
            break
            
    for attempt in range(3):
        try:
            summary = get_event_summary(monologue)
        except:
            summary = "Event Summary Format Save Error Occured"
        else:
            break

    uploadFirebase(user, num, frequency, summary)
    
    end_time = time.time()
    print(f"{end_time - start_time:.5f} sec")
    
    return {"frequency": frequency, "summary": summary}
        
@app.post("/gpt")
async def analysis(request: Request):
    start_time = time.time()
    body = await request.json()
    diary = body['text']
    
    result = get_word_frequency(diary)
    
    end_time = time.time()
    print(f"{end_time - start_time:.5f} sec")

    return result

#넣은 글을 Event, emotion/thought로 분리해서 요약해주는 API
@app.post("/summary")
async def analysis(request: Request):
    start_time = time.time()
    body = await request.json()
    text = body['text']
    
    result = get_event_summary(text)
    
    end_time = time.time()
    print(f"{end_time - start_time:.5f} sec")
    return result
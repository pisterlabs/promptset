###### 기본 정보 설정 단계 ######
import urllib3
import json
import openai
from fastapi import Request, FastAPI

# OpenAI API KEY
API_KEY = ""
openai.api_key = API_KEY

# Telegram API KEY
BOT_TOKEN = ""

###### 서버 생성 단계 #####

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "TelegramChatbot"}

@app.post("/chat/")
async def chat(request: Request):
    telegramrequest = await request.json()
    chatBot(telegramrequest)
    return {"message": "TelegramChatbot/chat"}

###### 기능 함수 구현 단계 ######
# 메세지 전송
def sendMessage(chat_id, text,msg_id):
    data = {
        'chat_id': chat_id,
        'text': text,
        'reply_to_message_id': msg_id
    }
    http = urllib3.PoolManager()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = http.request('POST',url ,fields=data)
    return json.loads(response.data.decode('utf-8'))

# 사진 전송
def sendPhoto(chat_id, image_url,msg_id):
    data = {
        'chat_id': chat_id,
        'photo': image_url,
        'reply_to_message_id': msg_id
    }
    http = urllib3.PoolManager()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    response = http.request('POST',url ,fields=data)
    return json.loads(response.data.decode('utf-8'))

# ChatGPT에게 질문/답변받기
def getTextFromGPT(messages):
    messages_prompt = [{"role": "system", "content": 'You are a thoughtful assistant. Respond to all input in 25 words and answer in korea'}]
    messages_prompt += [{"role": "system", "content": messages}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_prompt)
    system_message = response["choices"][0]["message"]
    return system_message["content"]

# DALLE.2 에게 질문/그림 URL 받기
def getImageURLFromDALLE(messages):   
    response = openai.Image.create(prompt=messages,n=1,size="512x512")
    image_url = response['data'][0]['url']
    return image_url

###### 메인 함수 구현 단계 #####

def chatBot(telegramrequest):
    
    result = telegramrequest
    if not result['message']['from']['is_bot']:

        # 메세지를 보낸 사람의 chat ID 
        chat_id = str(result['message']['chat']['id'])

        # 해당 메세지의 ID
        msg_id = str(int(result['message']['message_id']))
        # 만약 그림 생성을 요청하면
        if '/img' in result['message']['text']:
            prompt = result['message']['text'].replace("/img", "")
            # DALL.E 2로부터 생성한 이미지 URL 받기
            bot_response = getImageURLFromDALLE(prompt)
            # 이미지 텔레그램 방에 보내기
            print(sendPhoto(chat_id,bot_response, msg_id))
        # 만약 chatGPT의 답변을 요청하면
        if '/ask' in result['message']['text']:
            prompt = result['message']['text'].replace("/ask", "")
            # ChatGPT로부터 답변 받기
            bot_response = getTextFromGPT(prompt)
            # 답변 텔레그램 방에 보내기
            print(sendMessage(chat_id, bot_response,msg_id))
    
    return 0
import openai
import os



import requests
import json
#환경변수 설정후 IDE도구 다시 재시작
#print(os.getenv('OPENAI_API_KEY'))
print(dir(openai))
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
'''
#REST API로 요청  pip install openai가 필요없다(단,pip install requests)

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
url='https://api.openai.com/v1/chat/completions'
headers={"Content-Type":"application/json","Authorization": f"Bearer {OPENAI_API_KEY}"}
data= {
    "model":"gpt-3.5-turbo",
    "messages":[
        {"role":"system","content":"you are a kindergarten teacher"},
        {"role":"user","content":"What is Artificial Intelligence?"}
    ]}
res=requests.post(url=url,data=json.dumps(data),headers=headers)
#print(res)
#print(res.json())
response=res.json()
print(response["choices"][0]["message"]["content"])
'''

#open ai의 ChatGPT API 사용
model = 'gpt-3.5-turbo'
openai.api_key=OPENAI_API_KEY
def generate_chat(model,messages):
    response = openai.ChatCompletion.create(model=model,messages=messages)
    return response

messages = [
    {"role":"system","content":"you are a kindergarten teacher"},
    {"role":"user","content":"What is Artificial Intelligence?"}
]
response = generate_chat(model,messages)
#print(type(response))#<class 'openai.openai_object.OpenAIObject'>
#print(dir(response))
#print(response)
'''
{
  "id": "chatcmpl-7d7SYENiBg1lqlVftqWQU9kkP2DVp",
  "object": "chat.completion",
  "created": 1689557822,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Artificial Intelligence refers to the development of computer systems that can perform tasks that would typically require human intelligence. These systems are designed to analyze and interpret data, learn from experience, and make decisions or take actions based on that learning. AI can be found in various forms, such as speech recognition, image processing, natural language understanding, and problem-solving. In simple terms, it is the simulation of human intelligence in machines that are programmed to think and learn like humans."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "completion_tokens": 93,
    "total_tokens": 114
  }
}

'''
#print(response.choices[0].message['content'])
answer=response.choices[0].message['content']
'''
role은 assistant로 content은 이전 답으로 설정 추가
ChatGPT가 이전에 응답한 결과를 알 수 있도록하기 위함
왜냐하면 대화의 문맥을 유지하기 위함
아래 주석시 이전의 응답을 번역하지 않고 
질문을 번역한다(인공지능이란 무엇인가요?)
'''
#대화 문맥을 유지하기 위한 assitant로 이전 응답 설정
messages.append({"role":"assistant","content":answer})
#추가 질문 즉 영어로 대답은 이전 응답을 한국어로 번역하도록 추가 질의
messages.append({"role":"user","content":"한국어로 번역해 주세요"})
response = generate_chat(model,messages)
korean=response.choices[0].message
print(json.dumps(korean,ensure_ascii=False))
'''
"인공지능은 컴퓨터 과학의 한 분야로, 주로 인간의 지능을 필요로 하는 작업을 
수행할 수 있는 기계와 시스템을 만드는 것에 초점을 두고 있습니다. 
인공지능 시스템은 주변 환경을 인식하고 정보를 해석하며 이해하고, 
과거 경험으로부터 배우며, 추론하고 결정을 내리며, 인간과 커뮤니케이션할 수 있도록 설계됩니다.
 인공지능의 목표는 인간의 인지 능력을 복제하고 다양한 산업에서 효율성과 정확성을 
 향상시키기 위해 작업을 자동화하는 것입니다. 
이는 기계 학습, 자연어 처리, 컴퓨터 비전, 로봇 공학과 같은 분야를 포함합니다."
'''



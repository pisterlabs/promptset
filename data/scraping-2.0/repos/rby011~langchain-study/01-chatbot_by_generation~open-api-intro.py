import os
import openai 
from dotenv import load_dotenv

# API Key 관리
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# how to use gpt-4 series
# - https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4
# 
# billing
# - https://platform.openai.com/account/billing/overview
# 
# series :
# - gpt-4-1106-preview     : Trained up to Apr 2023, $0.01 → $0.03  / 1K token
# - gpt-4-vision-preview   : Trained Up to Apr 2023, $0.01 → $0.03  / 1K token
# - gpt-4                  : Trained Up to Sep 2021,  $0.03 → $0.06 / 1K token
# - gpt-3.5-turbo-1106	   : $0.0010 / 1K tokens	$0.0020 / 1K tokens
# - gpt-3.5-turbo-instruct : 	$0.0015 / 1K tokens	$0.0020 / 1K tokens
response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    
    messages =[
        {'role': 'system', 'content': 'you are an hepful assistant'},
        {'role': 'user', 'content':'what can you do?'},    
    ],

    # Random Level
    temperature = 1,    # random 수준 : 0 이 높음 / 1 이면 고정

    # Generation Length 
    n = 2,              # 답변 개수 , 잘 사용하지 않는다고 함
    stop = ',',         # chatgpt 는 한 글자씩 보면서 생성하는데 이 stop 문자열을 보면 생성을 멈춤
    max_tokens = 20,    # 1 token 이면 영어 기준 4 글자로 생각하면 편리
    
    # Advanced
    # frequency_penalty = # -2 ~ 2 사이 값. 반복적(음수이면 반복도 증가) 문장 생성에 대한 패널티
    # logit_bias =        # ★★ 특정 단어가 무조건 등장하게 만들도록 할 때, 토큰 값을 넣어야 함. 한국어는 까다롭고 영어는 괜찮음
)

for res in response.choices:
    print(res)
    
#
# function_call
#
# 아래와 같은 상황에서 외부 함수를 호출하도록 fall back 을 만듬
#
# I'm sorry, I am an AI language model and do not have real-time access to weather data. 
# #Please check a weather website or app for current weather conditions in Boston.
#
# https://wikidocs.net/
# - # Step 1: send the conversation and available functions to GPT
# - # Step 2: check if GPT wanted to call a function
# - # Step 4: send the info on the function call and function response to GPT
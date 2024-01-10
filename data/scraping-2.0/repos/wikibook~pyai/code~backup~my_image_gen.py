# 이미지 생성을 위한 모듈

import openai
import os
import textwrap

# OpenAI Chat Completions API를 이용해 한국어를 영어로 번역하는 함수
def translate_text_for_image(text):    
    # API 키 설정
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # 대화 메시지 정의
    user_content = f"Translate the following Korean sentences into English.\n {text}"
    messages = [ {"role": "user", "content": user_content} ]

    # Chat Completions API 호출
    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo", # 사용할 모델 선택 
                            messages=messages, # 전달할 메시지 지정
                            max_tokens=1000, # 최대 토큰 수 지정
                            temperature=0.8, # 완성의 다양성을 조절하는 온도 설정
                            n=1 # 생성할 완성의 개수 지정
                            )

    assistant_reply = response.choices[0].message['content'] # 첫 번째 응답 결과 가져오기
    
    return assistant_reply # 응답 반환

# OpenAI Chat Completions API를 이용해 이미지를 위한 상세 묘사를 생성하는 함수
def generate_text_for_image(text):
    # API 키 설정
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # 대화 메시지 정의
    user_content = f"Describe the following in 1000 characters to create an image.\n {text}"
    
    messages = [ {"role": "user", "content": user_content} ]

    # Chat Completions API 호출
    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo", # 사용할 모델 선택 
                            messages=messages, # 전달할 메시지 지정
                            max_tokens=1000, # 최대 토큰 수 지정
                            temperature=0.8, # 완성의 다양성을 조절하는 온도 설정
                            n=1 # 생성할 완성의 개수 지정
                        )
    
    assistant_reply = response.choices[0].message['content'] # 첫 번째 응답 결과 가져오기

    return assistant_reply # 응답 반환

# OpenAI Image API((DALL·E)를 이용해 영어 문장으로 이미지를 생성하는 함수
def generate_image_from_text(text_for_image, image_num=1, image_size="512x512"):    
    # API 키 설정
    
    shorten_text_for_image = textwrap.shorten(text_for_image, 1000) # 1,000자로 제한
    
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Image.create(prompt=shorten_text_for_image, n=image_num, size=image_size)
    
    image_urls = [] # 이미지 URL 리스트
    for data in response['data']:
        image_url = data['url'] # 이미지 URL 추출    
        image_urls.append(image_url)   
        
    return image_urls # 이미지 URL 리스트 반환

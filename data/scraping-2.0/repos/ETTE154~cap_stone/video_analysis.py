# %%
import cv2
import pysrt
import os
import pandas as pd
from datetime import datetime
import re
from datetime import timedelta

import base64
import requests
import json
from openai import OpenAI

import os
from dotenv import load_dotenv

client = OpenAI()

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# 이미지를 base64로 인코딩하는 함수
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# OpenAI API를 사용하여 이미지 설명을 얻는 함수
def get_image_description(api_key, base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return json.loads(response.text)['choices'][0]['message']['content']

def extract_barrier_free_subtitles(subs):
    return [sub for sub in subs if re.match(r'\[.*?\]', sub.text)]

def srttime_to_timedelta(subrip_time):
    return timedelta(hours=subrip_time.hours, minutes=subrip_time.minutes, 
                     seconds=subrip_time.seconds, milliseconds=subrip_time.milliseconds)

def extract_frames(video_path, subtitle_path, api_key):
    subs = pysrt.open(subtitle_path)
    barrier_free_subs = extract_barrier_free_subtitles(subs)

    cap = cv2.VideoCapture(video_path)

    folder_name = "Rationale"
    os.makedirs(folder_name, exist_ok=True)

    data = []

    for sub in barrier_free_subs:
        start_time = srttime_to_timedelta(sub.start)
        end_time = srttime_to_timedelta(sub.end)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time.total_seconds() * 1000)
        ret, frame = cap.read()

        if ret:
            time_str = f"{sub.start.hours:02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"
            image_path = f'{folder_name}/{time_str}.jpg'
            cv2.imwrite(image_path, frame)

            # 이미지를 base64로 인코딩하고 설명을 얻습니다
            base64_image = encode_image_to_base64(image_path)
            vision_rationale = get_image_description(api_key, base64_image)

            # 진동 패턴 및 강도 계산
            vibration_pattern = get_vibration_pattern(sub.text, vision_rationale)
            vibration_intensities = get_vibration_intensity(sub.text, vision_rationale, vibration_pattern)
            vibration_result = [vibration_pattern] + vibration_intensities

            data.append([sub.start, sub.end, sub.text, image_path, vision_rationale, vibration_result])

    cap.release()
    return pd.DataFrame(data, columns=['Start_Time', 'End_Time', 'Text_Rationale', 'Image_Path', 'Vision_Rationale', 'Vibration_Result'])

def get_vibration_pattern(text_rationale, vision_rationale):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.5,
        top_p=0.5,
        messages=[
            {"role": "system", "content": "영상의 배리어 프리 자막(텍스트)과 장면(이미지)에 대한 설명을 분석하여 삼차원 진동 조끼에 적합한 진동 패턴을 제안하세요."},
            {"role": "user", "content": f"장면의 배리어 프리 자막은 '{text_rationale}'"},
            {"role": "user", "content": f"장면에 대한 설명은 '{vision_rationale}'."},
            {"role": "system", "content": "삼차원 진동 조끼에서 발생시킬 수 있는 진동 패턴은 3가지 이다.0 : 진동이 증가하는 형태1 : 진동이 감소하는 형태2 : 일정한 크기를 가지는 진동 형태Q : 해당 장면은 어떤 형태의 진동을 발생시켜야 하나요?"},
            {"role": "user", "content": "'해당 장면은 진동이 증가하는 형태의 진동을 발생시켜야 합니다. 번호는 n입니다.' 형태로 알려주세요."},
        ]
    )

    # 정규 표현식 필터
    filter = re.compile(r'0|1|2')
    
    # response에서 content 속성 추출
    content = response.choices[0].message.content
    print(content)
    # content에서 숫자 찾기
    match = filter.findall(content)

    # 첫 번째 일치 항목을 정수로 변환
    return int(match[0]) if match else None

def get_vibration_intensity(text_rationale, vision_rationale, vibration_pattern):
    client = OpenAI()
    
    response = None
    
    # vibration_pattern이 리스트인 경우 첫 번째 요소를 사용
    if isinstance(vibration_pattern, list):
        vibration_pattern = int(vibration_pattern[0]) if vibration_pattern else None
    
    print("vibration_pattern :", vibration_pattern)
        
    if vibration_pattern == 0:
        v_pattern = "점점 강해지는"
    elif vibration_pattern == 1:
        v_pattern = "점점 약해지는"
    elif vibration_pattern == 2:
        v_pattern = "일정한 크기를 가지는"
        
    print("v_pattern : ", v_pattern)
    
    if v_pattern == "점점 강해지는" or v_pattern == "점점 약해지는":
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            top_p=0.5,
            # max_tokens=100,
            messages=[
                {"role": "system", "content": "삼차원 진동 조끼에 적합한 진동 강도를 제안하세요."},
                {"role": "user", "content": f"장면의 배리어 프리 자막은 '{text_rationale}'"},
                {"role": "user", "content": f"장면에 대한 설명은 '{vision_rationale}'."},
                {"role": "system", "content": f"Q : {v_pattern}의 형태를 가지는 진동을 발생할때, 진동의 최소 및 최대 강도(0 ~ 10 사이의값, 격한 장면일수록 큰 숫자)는 얼마인가요?"},
                {"role": "system", "content": "'최소 강도: min_num, 최대 강도: max_num' 형태로 알려주세요."}
            ]
        )
    elif  v_pattern == "일정한 크기를 가지는":
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            top_p=0.5,
            # max_tokens=100,
            messages=[
                {"role": "system", "content": "삼차원 진동 조끼에 적합한 진동 강도를 제안하세요."},
                {"role": "user", "content": f"장면의 배리어 프리 자막은 '{text_rationale}'"},
                {"role": "user", "content": f"장면에 대한 설명은 '{vision_rationale}'."},
                {"role": "system", "content": "Q : 일정한 크기를 가지는 진동을 발생할때, 진동 강도(0 ~ 10 사이의값, 격한 장면일수록 큰 숫자)는 얼마인가요?"},
                {"role": "system", "content": "'진동 강도는 n 입니다.' 형태로 알려주세요."}
            ]
        )


    # 정규 표현식 필터
    filter = re.compile(r'\d+')

    # response에서 content 속성 추출
    content = response.choices[0].message.content if response else ""
    print("content : ", content)
    # content에서 숫자 찾기
    matches = filter.findall(content)

    sorted_intensities = sorted([int(num) for num in matches]) if matches else []
    print("sorted_intensities : ", sorted_intensities)
    return sorted_intensities

df = extract_frames('내가죽던날_자막.mp4', '내가죽던날_자막.srt', api_key)
df.to_csv('extracted_frames_with_vibration.csv', index=False)

  # %%

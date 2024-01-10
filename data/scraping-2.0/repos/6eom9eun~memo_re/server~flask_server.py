from flask import Flask, request, jsonify
from transformers import pipeline
from IPython.display import Image
import openai
# from googletrans import Translator
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
import os

app = Flask(__name__)

api_key= "apiKey"

@app.route('/', methods=['POST'])
def generate_memory():
    text = request.form.get('text')
    image_directory = 'C:/Users/user/Desktop/memo_re/server/image' ### 플러터에서 받아온 이미지 저장할 폴더 설정할 것 ###
    os.makedirs(image_directory, exist_ok=True)

    # 이미지 파일을 받아오기
    if 'image' in request.files:
        image = request.files['image']
        image_path = os.path.join(image_directory, 'uploaded_image.jpg')
        image.save(image_path)
    else:
        # 이미지가 없는 경우에 대한 처리
        image_path = ''  

    metadata = get_image_metadata(image_path)
    image_description = generate_image_description(image_path)
    memory = generate_memory(metadata, image_description, text)
    translate = translate_to_english(memory)
    image_url = generate_image_from_memory(translate)

    # 이미지 URL과 메모리를 JSON 응답으로 반환
    response = {
        "memory": memory,
        "image_url": image_url
    }

    return jsonify(response)

# 이미지 설명을 생성하는 파이프라인 설정
def generate_image_description(image_path):

    if not image_path:
        return ""

    caption = pipeline("image-to-text")
    result = caption(image_path)
    image_description = result[0]['generated_text']
    return image_description


"""### 사진에서 날짜 추출"""

#사진에서 날짜 뽑기

def get_image_metadata(image_path):

    if not image_path:
        return ""

    try:
        # 이미지 열기
        image = Image.open(image_path)

        # 이미지의 Exif 메타데이터 읽어오기
        exif_data = image._getexif()

        if exif_data:
            metadata = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                metadata[tag_name] = value

            # DateTimeOriginal 또는 DateTime 추출
            if "DateTimeOriginal" in metadata:
                date_str = metadata["DateTimeOriginal"]
            elif "DateTime" in metadata:
                date_str = metadata["DateTime"]
            else:
                return "이미지에서 시간 데이터를 찾을 수 없습니다."

            # 날짜 시간 문자열을 파싱하여 datetime 객체 생성
            date_time = datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")

            # 시기 파악
            season = ""
            if date_time.month in (3, 4, 5):
                season = "봄"
            elif date_time.month in (6, 7, 8):
                season = "여름"
            elif date_time.month in (9, 10, 11):
                season = "가을"
            else:
                season = "겨울"

            # 시간 파악 (오전, 오후, 저녁)
            time_of_day = ""
            hour = date_time.hour
            if 5 <= hour < 12:
                time_of_day = "오전"
            elif 12 <= hour < 17:
                time_of_day = "오후"
            elif 17 <= hour < 20:
                time_of_day = "저녁"
            else:
                time_of_day = "밤"

            return {
                "DateTime": date_time,
                "Season": season,
                "TimeOfDay": time_of_day
            }
        else:
            return "이미지에서 상세정보를 찾을 수 없습니다."
    except Exception as e:
        return f"Error: {str(e)}"


# 추억 생성 함수

def generate_memory(metadata,image_description,keyword):


    if not image_description:
        image_description = ""
    if not metadata:
        metadata = ""


    # metadata에서 계절, 시간대, 날짜 데이터 추출
    if isinstance(metadata, str):
        season = metadata
        time_of_day = metadata
        date_time = metadata
    else:
        season = metadata.get("Season", "데이터 없음")
        time_of_day = metadata.get("TimeOfDay", "데이터 없음")
        date_time = metadata.get("DateTime", "데이터 없음")

# date_time 변수가 문자열인 경우 처리
    if isinstance(date_time, str):
        date_time_info = date_time
    else:
        date_time_info = date_time
        if hasattr(date_time, 'year'):
            date_time_info = f"{date_time.year}년 {date_time.month}월 {date_time.day}일"


    # keyword와 metadata 정보를 포함한 이야기 생성
    prompt = f"내 추억과 관련된 키워드: {keyword}, {image_description},{date_time_info}, {time_of_day}, {season}"
    print(prompt)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "넌 최고의 작가이자 시인이야. 내가 키워드를 주면, 거기에 관해서 공백 포함 300자 이내로 짧은 시를 만들어 줘.근데 키워드 각각의 이야기가 아니라 키워드가 전부 들어가는 시를 말해줘야 돼. 답변은 한번만 해주면 돼. 더 요구하지 마. 직접적인 년도는 말하지 마. 명심해 시로 출력해 줘야 돼."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # 출력 최대 길이 설정.
        api_key=api_key
    )

    # API 응답에서 생성된 추억 추출
    generated_memory = response.choices[0].message["content"].strip()

    return generated_memory

#번역


def translate_to_english(memory):  # 추억 번역기. 번역 안 하면, 이미지가 제대로 안 나옴.
    prompt = f"이 문장을 영어로 바꿔줘: {memory}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "넌 번역기야 내가 넣는 문장을 영어로 바꿔줘."},
            {"role": "user", "content": memory}
        ],
        max_tokens=1000,  # 출력 최대 길이 설정.
        api_key=api_key
    )

    trans = response.choices[0].message["content"].strip()

    return trans


# 이미지 생성 함수를 정의
def generate_image_from_memory(trans):
    response = openai.Image.create(
        prompt=trans,  # memory를 사용
        api_key=api_key,
        n=1,
        size="1024x1024",
        style="vivid"
    )
    image_url = response['data'][0]['url']
    return image_url



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
import requests
from bs4 import BeautifulSoup
import openai
import json
import pandas as pd
from flask import Flask,render_template,request,redirect, url_for,session,Response,send_file
import json
import requests
import torch
import os 
from diffusers import StableDiffusionPipeline
import io
from PIL import Image
import uuid
from MoverScore import mover_test_copy_update



# OpenAI API 인증
openai.api_key = ''

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/process', methods=['POST'])
def process():
    user_input_1=request.form['input1']
    user_input_2=request.form['input2']
    user_input_3=request.form['input3']
    processed_result = process_input(user_input_1,user_input_2,user_input_3)
    print(processed_result)
    session['processed_result'] = processed_result
    session['input_1'] = user_input_1
    session['input_2'] = user_input_2
    session['input_3'] = user_input_3
    
    return redirect(url_for('generate_image'))

@app.route('/generate_image')
def generate_image():
    # 세션에서 텍스트 가져오기
    processed_result = session.get('processed_result', '')
    print(processed_result)
    # 텍스트 요약 및 번역
    #summary_text = summary(processed_result)
    translated_text = get_translate(session['input_3']+"에 들어간 "+session['input_2'] )

    # 이미지 생성
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", variant='fp16')
    
    image = pipe(translated_text).images[0]

    image_path = save_image(image)
    session['image_path'] = image_path
    
    return redirect(url_for('result'))

@app.route('/result')
def result():
    input1 = session.get('input_1', '')
    input2 = session.get('input_2', '')
    input3 = session.get('input_3', '')
    processed_result = session.get('processed_result', '')
    
    return render_template('new_result.html', input1=input1, input2=input2,input3=input3, processed_result=processed_result)

@app.route('/image')
def image():
    image_path = session.get('image_path', '')
    if image_path:
        return send_file(image_path, mimetype='image/png')
    else:
        return "No image found", 404



def process_input(input_data_1,input_data_2,input_data_3):
    # 사용자로부터 입력 받기
    #user_input=input("들어가는 영화 첫번째 제목")
    #user_input_2=input("그 영화에 등장하는 인물")
    #user_input_3=input("배경이 되는 영화")

    #영화 정보 불러오기
    movie_info_1=movieData(input_data_1)
    movie_info_2=movieData(input_data_3)
    print(input_data_1)
    print(input_data_2)
    print(input_data_3)

    # 가져온 정보를 기반으로 챗지피티에게 문장 전달 및 답변 받기
    chat_input = f"너가 소설을 쓰는 작가라고 생각하고 {input_data_1}"+"에 등장하는 "+f"{input_data_2}"+"가 "+f"{input_data_3}"+"세계에 들어가면 어떻게 될지에 관한 소설만 적어줘. 최대한 길게 적어줘."+f"{input_data_1}"+"의 줄거리는 "+f"{movie_info_1}"+"이고 "+f"{input_data_3}"+"의 줄거리는 "+f"{movie_info_2}"+"야.\n"
    print(chat_input)
    chat_response = chat_with_gpt3(chat_input)

    # 챗지피티의 답변 출력
    print("챗지피티: ", chat_response)
    score=mover_test_copy_update.get_score(chat_response)
    print(score)
    if (score<0.8):
        chat_input = f"이전에 너가 써준 소설의 점수는 {score}야. 너가 소설을 쓰는 작가라고 생각하고 {input_data_1}"+"에 등장하는 "+f"{input_data_2}"+"가 "+f"{input_data_3}"+"세계에 들어가면 어떻게 될지에 관한 소설만 다시 적어줘. 최대한 길게 적어줘."+f"{input_data_1}"+"의 줄거리는 "+f"{movie_info_1}"+"이고 "+f"{input_data_3}"+"의 줄거리는 "+f"{movie_info_2}"+"야.\n"
        print(chat_input)
        chat_response = chat_with_gpt3(chat_input)
        score=mover_test_copy_update.get_score(chat_response)
        
    
    
    return chat_response

def movieData(query):
    key=''
    
    url='http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json.jsp?collection=kmdb_new&detail=Y'
    r=requests.post(url,data={'title':query,'ServiceKey':key,'createDts':'2018','createDts':'2018','val001':'2023','val002':'01'})
    
    movie_data = r.json()
    print(movie_data)
    
    
    max=-1
    
    for h in movie_data['Data'][0]['Result']:
        if max<int(h['runtime']):
            result=h['plot']
            max=int(h['runtime'])
    print(result)
    return result

def chat_with_gpt3(query):
    
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": query}]
)
    return response.choices[0].message.content
def get_translate(text):
    client_id = ""
    client_secret = ""

    data = {'text': text,
            'source': 'ko',
            'target': 'en'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id": client_id,
              "X-Naver-Client-Secret": client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if (rescode == 200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:", rescode)


def summary(text):
    client_id = ""
    client_secret = ""
    url = 'https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize'

    headers = {
        'Accept': 'application/json;UTF-8',
        'Content-Type': 'application/json;UTF-8',
        'X-NCP-APIGW-API-KEY-ID': client_id,
        'X-NCP-APIGW-API-KEY': client_secret
    }

    data = {
        "document": {
            "content": text
        },
        "option": {
            "language": "ko",
            "model": "general",
            "tone": 0,
            "summaryCount": 1
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('UTF-8'))
    rescode = response.status_code
    if (rescode == 200):
        print(response.text)
        return response.text
    else:
        print("Error : " + response.text)
        return "Error"


def save_image(image):
    """
    Saves a PIL Image object as a temporary file and returns the file path.
    
    :param image: A PIL Image object.
    :return: Path of the saved image file.
    """
    folder = "temp_images"
    if not os.path.exists(folder):
        os.makedirs(folder)  # 폴더가 없다면 생성

    filename = os.path.join(folder, f"{uuid.uuid4()}.png")
    image.save(filename)
    return filename


def main():
    # 사용자로부터 입력 받기
    user_input=input("들어가는 영화 첫번째 제목")
    user_input_2=input("그 영화에 등장하는 인물")
    user_input_3=input("배경이 되는 영화")

    #영화 정보 불러오기
    movie_info_1=movieData(user_input)
    movie_info_2=movieData(user_input_3)
    

    # 가져온 정보를 기반으로 챗지피티에게 문장 전달 및 답변 받기
    chat_input = f"{user_input}"+"에 등장하는 "+f"{user_input_2}"+"가 "+f"{user_input_3}"+"세계에 들어가면 어떻게 될지에 관한 소설을 적어줘. "+f"{user_input}"+"의 줄거리는 "+f"{movie_info_1}"+"이고 "+f"{user_input_3}"+"의 줄거리는 "+f"{movie_info_2}"+"야.\n"
    chat_response = chat_with_gpt3(chat_input)

    # 챗지피티의 답변 출력
    print("챗지피티: ", chat_response)

if __name__ == "__main__":
    app.run(debug=True)


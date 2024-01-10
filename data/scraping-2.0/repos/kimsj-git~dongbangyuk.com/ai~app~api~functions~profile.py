# Python3 샘플 코드 #

import xml.etree.ElementTree as ET
import requests
import openai
from dotenv import load_dotenv
import io
import os
import boto3
import uuid

load_dotenv()

saju_api = os.environ.get('SAJU_API')
openai_api = os.environ.get('OPENAI_API')
s3_bucket = os.environ.get('S3_BUCKET')
region_static = os.environ.get('REGION_STATIC')
aws_accesskey = os.environ.get('AWS_ACCESSKEY')
aws_secretkey = os.environ.get('AWS_SECRETKEY')


def saju(year, month, day):
    url = 'http://apis.data.go.kr/B090041/openapi/service/LrsrCldInfoService/getLunCalInfo'
    params ={'serviceKey' : saju_api, 'solYear' : year, 'solMonth' : month, 'solDay' : day}

    response = requests.get(url, params=params)

    root = ET.fromstring(response.content)

    # 딕셔너리 생성
    result = {}
    for child in root.iter():
        result[child.tag] = child.text

    prompt = 'There is a '

    if result['lunIljin'][0] == '갑' or result['lunIljin'][0] == '을':
        prompt += 'blue '

    elif result['lunIljin'][0] == '병' or result['lunIljin'][0] == '정':
        prompt += 'red '

    elif result['lunIljin'][0] == '무' or result['lunIljin'][0] == '기':
        prompt += 'yellow '

    elif result['lunIljin'][0] == '경' or result['lunIljin'][0] == '신':
        prompt += 'white '

    else:
        prompt += 'black '

    if result['lunIljin'][1] == '자':
        prompt += 'rat'

    elif result['lunIljin'][1] == '축':
        prompt += 'ox'

    elif result['lunIljin'][1] == '인':
        prompt += 'tiger'

    elif result['lunIljin'][1] == '묘':
        prompt += 'rabbit'

    elif result['lunIljin'][1] == '진':
        prompt += 'dragon'

    elif result['lunIljin'][1] == '사':
        prompt += 'snake'

    elif result['lunIljin'][1] == '오':
        prompt += 'horse'

    elif result['lunIljin'][1] == '미':
        prompt += 'lamb'

    elif result['lunIljin'][1] == '신':
        prompt += 'monkey'

    elif result['lunIljin'][1] == '유':
        prompt += 'rooster'

    elif result['lunIljin'][1] == '술':
        prompt += 'dog'

    elif result['lunIljin'][1] == '해':
        prompt += 'pig'

    prompt += ' and behind it is a big'

    #  일주 보내기

    # 연주
    if result['lunSecha'][0] == '갑' or result['lunSecha'][0] == '을':
        prompt += ' blue '

    elif result['lunSecha'][0] == '병' or result['lunSecha'][0] == '정':
        prompt += ' red '

    elif result['lunSecha'][0] == '무' or result['lunSecha'][0] == '기':
        prompt += ' yellow '

    elif result['lunSecha'][0] == '경' or result['lunSecha'][0] == '신':
        prompt += ' white '

    else:
        prompt += ' black '

    if result['lunSecha'][1] == '자':
        prompt += 'rat'

    elif result['lunSecha'][1] == '축':
        prompt += 'ox'

    elif result['lunSecha'][1] == '인':
        prompt += 'tiger'

    elif result['lunSecha'][1] == '묘':
        prompt += 'rabbit'

    elif result['lunSecha'][1] == '진':
        prompt += 'dragon'

    elif result['lunSecha'][1] == '사':
        prompt += 'snake'

    elif result['lunSecha'][1] == '오':
        prompt += 'horse'

    elif result['lunSecha'][1] == '미':
        prompt += 'lamb'

    elif result['lunSecha'][1] == '신':
        prompt += 'monkey'

    elif result['lunSecha'][1] == '유':
        prompt += 'rooster'

    elif result['lunSecha'][1] == '술':
        prompt += 'dog'

    elif result['lunSecha'][1] == '해':
        prompt += 'pig'

    prompt += ' like the background, digital art, trending on artstation'

    # 결과 출력

    # prompt = ''
    # # 연주
    # if result['lunSecha'][0] == '갑' or result['lunSecha'][0] == '을':
    #     prompt += 'A Blue '
    #
    # elif result['lunSecha'][0] == '병' or result['lunSecha'][0] == '정':
    #     prompt += 'A Red '
    #
    # elif result['lunSecha'][0] == '무' or result['lunSecha'][0] == '기':
    #     prompt += 'A Yellow '
    #
    # elif result['lunSecha'][0] == '경' or result['lunSecha'][0] == '신':
    #     prompt += 'A White '
    #
    # else:
    #     prompt += 'A Black '
    #
    #
    # if result['lunSecha'][1] == '자':
    #     prompt += 'rat'
    #
    # elif result['lunSecha'][1] == '축':
    #     prompt += 'ox'
    #
    # elif result['lunSecha'][1] == '인':
    #     prompt += 'tiger'
    #
    # elif result['lunSecha'][1] == '묘':
    #     prompt += 'rabbit'
    #
    # elif result['lunSecha'][1] == '진':
    #     prompt += 'dragon'
    #
    # elif result['lunSecha'][1] == '사':
    #     prompt += 'snake'
    #
    # elif result['lunSecha'][1] == '오':
    #     prompt += 'horse'
    #
    # elif result['lunSecha'][1] == '미':
    #     prompt += 'lamb'
    #
    # elif result['lunSecha'][1] == '신':
    #     prompt += 'monkey'
    #
    # elif result['lunSecha'][1] == '유':
    #     prompt += 'rooster'
    #
    # elif result['lunSecha'][1] == '술':
    #     prompt += 'dog'
    #
    # elif result['lunSecha'][1] == '해':
    #     prompt += 'pig'
    #
    # prompt += ' with '
    #
    # #  일주 보내기
    # if result['lunIljin'][0] == '갑' or  result['lunIljin'][0] == '을':
    #     prompt += 'A Blue '
    #
    # elif result['lunIljin'][0] == '병' or  result['lunIljin'][0] == '정':
    #     prompt += 'A Red '
    #
    # elif result['lunIljin'][0] == '무' or  result['lunIljin'][0] == '기':
    #     prompt += 'A Yellow '
    #
    # elif result['lunIljin'][0] == '경' or  result['lunIljin'][0] == '신':
    #     prompt += 'A White '
    #
    # else:
    #     prompt += 'A Black '
    #
    #
    # if result['lunIljin'][1] == '자':
    #     prompt += 'rat'
    #
    # elif result['lunIljin'][1] == '축':
    #     prompt += 'ox'
    #
    # elif result['lunIljin'][1] == '인':
    #     prompt += 'tiger'
    #
    # elif result['lunIljin'][1] == '묘':
    #     prompt += 'rabbit'
    #
    # elif result['lunIljin'][1] == '진':
    #     prompt += 'dragon'
    #
    # elif result['lunIljin'][1] == '사':
    #     prompt += 'snake'
    #
    # elif result['lunIljin'][1] == '오':
    #     prompt += 'horse'
    #
    # elif result['lunIljin'][1] == '미':
    #     prompt += 'lamb'
    #
    # elif result['lunIljin'][1] == '신':
    #     prompt += 'monkey'
    #
    # elif result['lunIljin'][1] == '유':
    #     prompt += 'rooster'
    #
    # elif result['lunIljin'][1] == '술':
    #     prompt += 'dog'
    #
    # elif result['lunIljin'][1] == '해':
    #     prompt += 'pig'
    #
    # prompt += ' in front of it, digital art, trending on artstation'

    image_url = create_image(prompt)

    return image_url


def create_image(prompt):
    openai.api_key = openai_api

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )

    image_url = response['data'][0]['url']

    response = requests.get(image_url)
    img = io.BytesIO(response.content)

    client_s3 = boto3.client('s3',
                             aws_access_key_id=aws_accesskey,
                             aws_secret_access_key=aws_secretkey,
                             )

    file_name = str(uuid.uuid4()) + ".png"

    # client_s3.upload_file(img, s3_bucket, 'images/' + file_name)
    client_s3.put_object(
        Body=img,
        Bucket=s3_bucket,
        Key='images/' + file_name,
        ContentType='image/png'
    )

    s3_file_path = f'https://{s3_bucket}.s3.{region_static}.amazonaws.com/images/{file_name}'

    return s3_file_path
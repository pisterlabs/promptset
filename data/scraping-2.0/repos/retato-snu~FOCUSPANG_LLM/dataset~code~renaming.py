import os
from unittest import result
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()
dataset_path = "/home/latent14/Documents/dataset.json"
result_path = "/home/latent14/Documents/dataset1.json"

openai.api_key = os.environ.get('OPENAI_API_KEY')

given_name = ['민준', '서준', '도윤', '예준', '시우', '하준', '지호', '주원', '지후', '준우', '준서', '도현', '건우', '현우', '우진', '지훈', '선우', '유준', '서진', '연우', '은우', '민재', '현준', '시윤', '정우', '이준', '승우', '윤우', '지우', '지환', '승현', '유찬', '준혁', '수호', '승민', '시후', '진우', '민성', '수현', '준영', '지원', '이안', '재윤', '시현', '태윤', '한결', '지안', '동현', '윤호', '시원', '은찬', '시온', '민우', '재원', '민규', '지한', '서우', '은호', '재민', '민찬', '우주', '우빈', '하율', '준호', '지율', '율', '성민', '하진', '승준', '성현', '재현', '현서', '민호', '태민', '준', '예성', '지민', '지성', '윤재', '태현', '민혁', '하람', '규민', '성준', '로운', '하민', '윤성', '정민', '태양', '이현', '은성', '예찬', '준수', '도훈', '준희', '다온', '민석', '주안', '건', '주호','서윤', '서연', '지우', '하윤', '서현', '하은', '민서', '지유', '윤서', '채원', '수아', '지민', '지아', '지윤', '다은', '은서', '예은', '지안', '소율', '서아', '예린', '하린', '수빈', '소윤', '예원', '지원', '유나', '시은', '채은', '유진', '윤아', '예나', '가은', '시아', '아린', '예서', '서영', '연우', '예진', '민지', '주아', '하율', '수민', '다인', '수연', '유주', '아윤', '연서', '서우', '아인', '시연', '서은', '다연', '채윤', '나은', '서율', '하연', '나윤', '지율', '현서', '서하', '서진', '유빈', '다현', '채아', '예지', '수현', '소은', '사랑', '나연', '지은', '시현', '예빈', '민주', '은채', '세아', '윤지', '소연', '다윤', '지현', '주하', '지수', '승아', '소민', '혜원', '다온', '채린', '하영', '민아', '나현', '서희', '세은', '아영', '도연', '규리', '이서', '가윤', '유하', '아현', '연아']
first_name = ['김', '이', '박', '정', '최', '조', '강', '윤', '장', '임', '신', '유', '한', '오', '서', '전', '권', '황', '안', '송', '홍', '양', '고', '문', '손', '배', '백', '허', '노', '남', '심', '하', '주', '구', '곽', '성', '차', '우', '진', '민', '류', '나', '지', '엄', '변', '채', '원', '방', '천', '공', '현', '함', '여', '염', '석', '추', '도', '소', '설', '선', '마', '길', '연', '위', '표', '명', '기', '반', '라', '왕', '금', '옥', '육', '인', '맹', '제', '모', '남', '탁', '국', '어', '경', '은', '편', '용', '예', '봉', '사', '부', '황', '가', '복', '태', '목', '형', '피', '두', '감', '호', '제']
names = []

for i in range(len(given_name)):
    for j in range(len(first_name)):
        names.append(first_name[j]+given_name[i])

print(len(names))

with open(dataset_path, 'r') as json_file:
    data1 = json.load(json_file)


dataset = []
for i in range(len(data1)):
    data = data1[i]
    category = data["category"]
    if category.find("student") == -1:
        name = data["target"].split('_')[1][:2]
    else:
        name = data["target"][-3:]
    new_name = names[i]
    data = {
        "instruction": data["instruction"].replace(name,new_name),
        "input": data["input"],
        "output": data["output"].replace(name,new_name),
        "target": data["target"].replace(name,new_name),
        "category": data["category"]
    }
    dataset.append(data)

with open(result_path, "w", encoding='utf-8') as json_file:
    json.dump(dataset, json_file, indent = 4, ensure_ascii=False)
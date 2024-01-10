import mysql.connector
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import re
from openai import OpenAI
import json
from datetime import date

def default_converter(o):
    if isinstance(o, date):
        return o.strftime("%Y-%m-%d")  # 날짜를 문자열로 변환
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# .env 파일 로드 및 환경 변수 로드
load_dotenv()
host = os.getenv('MYSQL_HOST')
port = int(os.getenv('MYSQL_PORT'))
user = os.getenv('MYSQL_USER')
password = os.getenv('MYSQL_PASSWORD')
database = os.getenv('MYSQL_DATABASE')

# 데이터베이스 연결 설정
conn = mysql.connector.connect(
    host=host,
    user=user,
    passwd=password,
    database=database
)

cursor = conn.cursor()

# 데이터 조회 쿼리
query = "SELECT * FROM Articles"
cursor.execute(query)

# 컬럼 이름 가져오기
column_names = [column[0] for column in cursor.description]

# 데이터를 JSON 형식으로 변환
rows = cursor.fetchall()
data = [dict(zip(column_names, row)) for row in rows]

# JSON 파일로 저장
with open('data.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=4, default=default_converter)

# 연결 종료
cursor.close()
conn.close()

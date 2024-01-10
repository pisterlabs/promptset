from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup as bs
from data import SecondhandPost

## 당근 스크래핑 클래스
class Dangn:
    def __init__(self, cursor):
        self.category_nums = []
        self.name = 'dangn'



        ## DB에 저장된 IT 기기명 리스트
        sql_query = 'select device_name from device'
        cursor.execute(sql_query)
        self.device_list = cursor.fetchall()





    ######## 공통 함수

    ## 게시글 리스트에서 제품의 링크를 추출하는 함수
    def extract_link(self):
        pass

    ## 게시글의 정보를 추출하여 데이터 클래스로 반환하는 함수
    def extract_text(self):
        pass

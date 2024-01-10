from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd


import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # CORS 라이브러리 추가
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain_experimental.agents.agent_toolkits import create_csv_agent

from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.document_loaders import DataFrameLoader
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import AgentType
import json
import pymysql
import openai
app = Flask(__name__)
CORS(app)  # CORS 활성화



# 기존 코드

os.environ['OPENAI_API_KEY'] = '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'
os.environ['SERPAPI_API_KEY'] = 'eb49e541facea5012be6f8729c31c1c0a7720be3521e14153f96c77adff576e4'

df = pd.read_csv('C:/langchain/poet/news_new.csv')  # 경로 구분자를 '/'로 변경
def summarize(content):
    return content.split('.')[0] if pd.notnull(content) else ''
def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)


df['summary'] = df['news_content'].apply(summarize)

# create_csv_agent 함수를 통해 agent 변수 생성
agent1 = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    'C:/langchain/poet/company2.csv',  # 경로 구분자를 '/'로 변경
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    reduce_k_below_max_tokens=True
)


llm = ChatOpenAI(model="gpt-4-1106-preview")
ddg_search = DuckDuckGoSearchResults()

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

prompt_template = "Summarize the following content: {content}"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)
tools = [ddg_search, web_fetch_tool, summarize_tool]

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent2 = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True
)
llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent3 = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
load_dotenv()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}





llm = ChatOpenAI(model="gpt-4-1106-preview")



tools = [ddg_search, web_fetch_tool, summarize_tool]

agent4 = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

class CommaSeparatedListOutputParser(BaseOutputParser):
    """LLM 아웃풋에 있는 ','를 분리해서 리턴하는 파서."""


    def parse(self, text: str):
        return text.strip().split(", ")


# ChatOpenAI 객체 생성
llm = ChatOpenAI(model="gpt-4")


template = """"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

agent5 = LLMChain(
    llm=ChatOpenAI(model="gpt-4-1106-preview"),
    prompt=chat_prompt
)

template = "너는 마케팅전문가야. 콜라보 마케팅 사례 1부터 10까지 알려줘."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

# chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()


# LLMChain을 이용한 새로운 체인 설정
chain = LLMChain(llm=ChatOpenAI(), prompt=chat_prompt)





# Flask 엔드포인트 추가
# fetch로 연결하기 위해선 c:\eGovFrame-4.0.0\workspace.edu\SpringMVC13\src\main\java\kr\spring\config\WebConfig.java 코드가 필요
@app.route('/')
def index():
    return render_template('collaboration/request.jsp')

@app.route('/case',methods=['POST'])
def case():
    industry = request.json['industry']
    try:
        marketing = agent5.invoke({"text": industry+ "업종에서 최근 진행한 콜라보마케팅 사례10개를 python 리스트형태(각각의 콜라보 마케팅 사례10개가 문자열 형태로 포함)로 만들어줘"})
        marketing = marketing['text']
        

        # JSON 파트 추출
        list_start = marketing.find('[') + 1 
        list_end = marketing.rfind(']') 
        list_part = marketing[list_start:list_end]

        print(list_part)
        return list_part
    except Exception as e:
        return "fail"

@app.route('/ask_question', methods=['POST'])
def ask_question():
    industry = request.json['industry']
    question = request.json['question']  # 변경된 부분
    req_num = request.json['req_num']  # 변경된 부분
    #list_part = request.json['list_part']
    try:
        print("의뢰번호" + req_num)
        print(question)
       



        result = agent1.run(question+ ", 라는 내용에서  추천하는 기업 분야 3개 각 분야에 대표 기업 이름 3개, 분야 추천 근거를 5문장이상, 각 분야별 적합한 마케팅 전략을 json형태('recommendations'키가 있고 하위 키는 'industry','companies','reason','solution'인 형태)로 만들어줘.")
        
        # JSON 파트 추출
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_part = result[json_start:json_end]

        # 파싱

        # JSON 데이터 파싱
        data = json.loads(json_part)
        print(len(data['recommendations']))
        
        # 로컬 mysql과 커넥션 수행
        conn = pymysql.connect(host='project-db-stu3.smhrd.com', port=3307, user='Insa4_Spring_final_1', password='aischool1', db='Insa4_Spring_final_1_2', charset='utf8')
        curs = conn.cursor()

        
        for i in range(len(data['recommendations'])):
            print(i)
            industry = data['recommendations'][i]['industry']
            company1 = data['recommendations'][i]['companies'][0]
            company2 = data['recommendations'][i]['companies'][1]
            company3 = data['recommendations'][i]['companies'][2]
            reason = data['recommendations'][i]['reason']
            marketing_strategy = data['recommendations'][i]['solution']
            




            try:
                sql = "INSERT INTO tb_solution(req_num, sol_content, reco_industry, company1, company2, company3, marketing_strategy)  VALUES (%s, %s, %s, %s, %s, %s, %s)"
            
                #values = (req_num, reason, industry,company1, company2, company3,marketing_strategy,list_part)
                values = (req_num, reason, industry,company1, company2, company3,marketing_strategy)
                curs.execute(sql, values)
                print("행 삽입")
            except pymysql.IntegrityError as e:
                # 중복된 primary key가 발생한 경우 예외 처리
                print("중복된 primary key가 발생했습니다. 다음 행으로 넘어갑니다.")
                 
            except pymysql.Error as e:
                # 기타 DB 에러 처리
                print("DB 에러 발생:", e)
            

            

        # DB의 변화 저장
        conn.commit()
        conn.close()
        
        
        
        
        
        return "success"
    except Exception as e:
        return "fail"

if __name__ == '__main__':
    app.run(debug=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
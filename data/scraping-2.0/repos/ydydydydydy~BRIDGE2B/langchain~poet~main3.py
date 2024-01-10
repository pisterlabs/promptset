import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # CORS 라이브러리 추가
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 기존 코드
load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'
df = pd.read_csv('C:/langchain/poet/company2.csv')  # 경로 구분자를 '/'로 변경

# create_csv_agent 함수를 통해 agent 변수 생성
agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    'C:/langchain/poet/company2.csv',  # 경로 구분자를 '/'로 변경
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

# Flask 엔드포인트 추가
# fetch로 연결하기 위해선 c:\eGovFrame-4.0.0\workspace.edu\SpringMVC13\src\main\java\kr\spring\config\WebConfig.java 코드가 필요
@app.route('/')
def index():
    return render_template('collaboration/request.jsp')

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']  # 변경된 부분
    try:
        result = agent.run(question)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
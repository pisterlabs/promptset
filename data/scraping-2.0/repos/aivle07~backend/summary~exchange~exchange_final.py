import FinanceDataReader as fdr
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()

# 환율 라이브러리 가져와서 ChatOpenAI와 연결된 에이전트를 생성
def get_exchange():
    df1, df2 = get_exchange_data()
    
    # ChatOpenAI 에이전트 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0613'), 
        [df1, df2], 
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent

def get_exchange_data():
    df1 = fdr.DataReader('USD/KRW')
    df2 = fdr.DataReader('JPY/KRW')
    df1['Currency'] = '달러'
    df2['Currency'] = '엔화'
    df1['exchange'] = '환율'
    df2['exchange'] = '환율'
    return df1,df2

# ChatOpenAI 에이전트에게 질문을 던져서 대답하는 함수
def get__exchange_answer(agent, question):
    result = agent.run(question) 
    
    return result

if __name__ == '__main__':
    agent = get_exchange() 
    question = question = input("질문을 입력하세요: ")
    result = get__exchange_answer(agent, question)
    print(result)


# 그래프 용 최근 5일 환율
# import pandas as pd

# df1 = fdr.DataReader('USD/KRW', "2023-12-01") 
# df2 = fdr.DataReader('JPY/KRW', "2023-12-01") 
# df1['Currency'] = '달러'
# df2['Currency'] = '엔화'

# df_dollar = df1[-5:]['Close']
# df_yen = df2[-5:]['Close']
# print(df_dollar)
# print(df_yen)

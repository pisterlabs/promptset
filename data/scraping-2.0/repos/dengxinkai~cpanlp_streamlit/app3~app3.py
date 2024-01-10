import streamlit as st
import asyncio
import faiss
import numpy as np
import pandas as pd
import re
import time
import math
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Union,Callable,Dict, Optional, Any, Tuple
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.schema import BaseLanguageModel,Document
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.retrievers import TimeWeightedVectorStoreRetriever
import boto3
st.set_page_config(
    page_title="数字人",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app1/shuziren.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': '社科实验数字人'
    }
)
@st.cache_resource
def load_digitalaws():
    dynamodb = boto3.client(
        'dynamodb',
        region_name="ap-southeast-1", 
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    table_name = 'digit_human1'
    response = dynamodb.scan(
        TableName=table_name
    )
    items = response['Items']
    dfaws = pd.DataFrame(items)
    return dfaws
@st.cache_data(persist="disk")
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
with st.sidebar:
    st.image("https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app1/shuziren.jpg")
    with st.expander("👇 :blue[**第一步：输入 OpenAI API 密钥**]"):
        if 'input_api' in st.session_state:
            st.text_input(st.session_state["input_api"], key="input_api",label_visibility="collapsed")
        else:
            st.info('请先输入正确的openai api-key')
            st.text_input('api-key','', key="input_api",type="password")
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3,help="用于控制生成文本随机性和多样性的参数。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="用于控制生成文本中单词重复频率的技术。数值越大，模型对单词重复使用的惩罚就越严格，生成文本中出现相同单词的概率就越低；数值越小，生成文本中出现相同单词的概率就越高。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3,help="用于控制语言生成模型生成文本时对输入提示的重视程度的参数。presence_penalty的值较低，模型生成的文本可能与输入提示非常接近，但缺乏创意或原创性。presence_penalty设置为较高的值，模型可能生成更多样化、更具原创性但与输入提示较远的文本。")
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.3,help="用于控制生成文本的多样性，较小的top_p值会让模型选择的词更加确定，生成的文本会更加一致，而较大的top_p值则会让模型选择的词更加多样，生成的文本则更加多样化。")
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0)
    USER_NAME = st.text_input("请填写创数人姓名","Person", key="user_name")
agent_keys = [key for key in st.session_state.keys() if key.startswith('agent')]   
if st.button('刷新页面'):
    st.experimental_rerun()
    st.cache_data.clear()
if agent_keys:
    do_traits=[]
    with st.expander("当前数字人："):
        for i,key in enumerate(agent_keys):
            y=st.session_state[key]
            col1, col2= st.columns([1, 1])
            with col1:
                do_traits.append(y.traits)
                person=y.traits+"的人"
                st.write(person)
            with col2:
                if st.button('删除',key=f"del_{key}"):
                    del st.session_state[key]
                    st.experimental_rerun()
        df = pd.DataFrame({
                        '特征': do_traits
                    })

        st.dataframe(df, use_container_width=True)
        if st.button('删除所有数字人',key=f"delete_all"):
            for i,key in enumerate(agent_keys):
                del st.session_state[key]
            st.experimental_rerun()
        csv = convert_df(df)
        st.download_button(
           "下载数字人",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )

else:
    st.warning("当前不存在数字人") 
tab1, tab3 = st.tabs(["数字人创建", ":blue[**社科调查**]"])
LLM = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        openai_api_key=st.session_state.input_api
    ) 
agents={}
class GenerativeAgent(BaseModel):
    traits: str
    llm: BaseLanguageModel
    class Config:
        arbitrary_types_allowed = True
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]
  
    def generate_reaction(
        self,
        observation: str,
    ) -> str:
        prompt = PromptTemplate.from_template(
            "You are {traits} and must only give {traits} answers."
                +"\nQuestion: {observation}"
                +"\n{traits}answer:中文回答"       
        )
 
        kwargs = dict(
                      traits=self.traits,
                     
                      observation=observation
                    )
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = action_prediction_chain.run(**kwargs)
        return result.strip()
  
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    new_message = f"{message}"
    return (agent.traits+"的人觉得："+agent.generate_reaction(new_message))

with tab1:
    with st.expander("单个创建"):
        traits = st.text_input('特征','既内向也外向，渴望成功', key="name_input1_4",help="性格特征，不同特征用逗号分隔")
        if st.button('创建',help="创建数字人"):
            global agent1
            global agentss
            agent1 = GenerativeAgent(
              traits=traits,
              llm=LLM
             )            
            st.session_state[f"agent_{traits}"] = agent1
            st.experimental_rerun()
    uploaded_file = st.file_uploader("csv文件上传批量建立", type=["csv"],help="csv格式：特征")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
        for index, row in data.iterrows():
      
            traits = row['特征']
      
            st.session_state[f"agent_{traits}"]  = GenerativeAgent(
                  traits=traits,
                  llm=LLM,
               
                 )

with tab3:            
    if agent_keys:
        do_inter_name=[]
        do_inter_quesition=[]
        do_inter_result=[]
        interws = []
        interview = st.text_input('采访','你怎么看待', key="interview")
        if st.button('全部采访',help="全部采访",type="primary",key="quanbu"):
            with st.expander("采访结果",expanded=True):
                start_time = time.time()
                with get_openai_callback() as cb:
                    async def interview_all_agents(agent_keys, interview):
                        tasks = []
                        for key in agent_keys:
                            task = asyncio.create_task(interview_agent_async(st.session_state[key], interview))
                            tasks.append(task)
                        results = await asyncio.gather(*tasks)
                        for key, inter_result in zip(agent_keys, results):
                            st.success(inter_result)
                            do_inter_name.append(st.session_state[key].traits)
                            do_inter_quesition.append(interview)
                            do_inter_result.append(inter_result)
                        return do_inter_name,do_inter_quesition, do_inter_result
                    async def interview_agent_async(agent, interview):
                        inter_result = await asyncio.to_thread(interview_agent, agent, interview)
                        return inter_result
                    do_inter_name, do_inter_quesition,do_inter_result = asyncio.run(interview_all_agents(agent_keys, interview))
                    st.success(f"Total Tokens: {cb.total_tokens}")
                    st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.success(f"Completion Tokens: {cb.completion_tokens}")
                    st.success(f"Total Cost (USD): ${cb.total_cost}")
                end_time = time.time()
                st.write(f"采访用时：{round(end_time-start_time,2)} 秒")
        df_inter = pd.DataFrame({
                    '被采访人':do_inter_name,
                    '采访问题':do_inter_quesition,
                    '采访结果': do_inter_result,
                })
        if len(df_inter) > 1:
            question = df_inter.loc[0, '采访问题']
            merged_results = ''.join(df_inter['采访结果'])
            summary_template = """用统计学的方法根据上述回答{answer},对关于{question}问题的回答进行总结，并分析结论是否有显著性?"""
            summary_prompt = PromptTemplate(template=summary_template, input_variables=["answer", "question"])
            llm_chain = LLMChain(prompt=summary_prompt, llm=LLM)
            st.write(llm_chain.predict(answer=merged_results, question=question))
        with st.expander("采访记录"):
            st.dataframe(df_inter, use_container_width=True)
            csv_inter = convert_df(df_inter)
            st.download_button(
               "下载采访记录",
               csv_inter,
               "file.csv",
               "text/csv",
               key='download-csv_inter'
            )
if st.button('aws',key="aws"):

    dfaws = load_digitalaws()
    for index, row in dfaws.iterrows():
        traits = row['特征'].get('S', '')              
        st.session_state[f"agent_{traits}"]  = GenerativeAgent(
              traits=traits,
           
              llm=LLM,
           
             )









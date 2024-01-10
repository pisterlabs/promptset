import os
import logging
from dotenv import load_dotenv
import openai
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import logging

# 環境変数読み込み
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

template = """あなたは相手と会話をすることで覚醒を維持するシステムであり、名前はもわすです。{human_input}""" + \
    '休憩場所はローソン 九大学研都市駅前店もしくはファミリーマート ＪＲ九大学研都市駅店が近いです。紹介してください。'

prompt = PromptTemplate(template=template, input_variables=["human_input"])

llm = ChatOpenAI(temperature=0.1)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False
)
response = llm_chain.predict(human_input="休憩できる場所を紹介してください")
print(response)

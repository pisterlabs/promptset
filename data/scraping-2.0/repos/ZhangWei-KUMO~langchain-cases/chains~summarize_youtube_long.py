# pip install youtube-transcript-api
# pip3 install pytube
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv
load_dotenv('.env')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3600, chunk_overlap=600)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 这里的url是youtube的视频地址，注意，这里的视频必须是有英文字幕的，否则无法进行处理
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=i-wpzS9ZsCs',add_video_info=True)
data = loader.load()
if(len(data) == 0):
    print("没有找到英文字幕")
    exit()
texts = text_splitter.split_documents(data)
print(texts)
# print(f"本视频的博主: {data[0].metadata['author']},标题《 {data[0].metadata['title']}》,总时长为{data[0].metadata['length']}秒")
llm = OpenAI(temperature=0.7,openai_api_key=OPENAI_API_KEY)
# # 对于长文本，我们使用map_reduce的方式来进行处理
# chain = load_summarize_chain(llm, chain_type="map_reduce",verbose=False)
# summarize_en = chain.run(texts)
# chat = ChatOpenAI(temperature=0)
# template = "You are a helpful assistant that translates {input_language} to {output_language}."
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# chain = LLMChain(llm=chat, prompt=chat_prompt)
# summarize_cn = chain.run(input_language="English", output_language="Chinese", text=summarize_en, max_tokens=4000)
# print(summarize_cn)
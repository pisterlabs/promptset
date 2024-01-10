
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 最简单的chain架构
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="为生产{product}的公司取一个亮眼中文名字：",
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("电脑"))


# 在chain中加入memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """你是聊天机器人小瓜，你可以和人类聊天。

{memory}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["memory", "human_input"], template=template
)

# memory = ConversationBufferMemory(memory_key="memory")

memory = ConversationSummaryMemory(llm=OpenAI(
    temperature=0), buffer="以中文表示", memory_key="memory") # 通过memory key 将之前的对话信息传入到prompt中

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

print(llm_chain.run("你是谁？"))
print("---------------")
output = llm_chain.run("我刚才问了你什么，你是怎么回答的？")
print(output)


# retrieval QA
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("ftMLDE-2021.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=60,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents([pages[i].page_content for i in range(2,20)])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0), #语言模型
    chain_type="stuff",  # prompt的组织方式，后面细讲
    retriever=db.as_retriever() # 检索器
)

query = "what's the process of MLDE"
response = qa_chain.run(query)
print(response)


# details of RetrievlaQA chain
print('================qa_chain===============')
print(qa_chain)
print('======combine_documents_chain==========')
print(qa_chain.combine_documents_chain.document_prompt)
print('==============llm_chain================')
print(qa_chain.combine_documents_chain.llm_chain.prompt.template)




# 常见的chain类型之Sequential
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
name_prompt = PromptTemplate(
    input_variables=["query"],
    template="从给定句子中提取出完整地址：{query}\n直接输出结果。",
)

name_chain = LLMChain(llm=llm, prompt=name_prompt)

slogan_prompt = PromptTemplate(
    input_variables=["address"],
    template="将'{address}'翻译成英文\n直接输出结果。",
)

slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)

overall_chain = SimpleSequentialChain(
    chains=[name_chain, slogan_chain], verbose=True)

print(overall_chain.run("收件地址北京市朝阳区东方东路19号"))

# 常见Chain类型之Transform
import re
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 例如：发给OpenAI之前，把用户隐私数据抹掉
def anonymize(inputs: dict) -> dict:
    text = inputs["text"]
    t = re.compile(
        r'1(3\d|4[4-9]|5[0-35-9]|6[67]|7[013-8]|8[0-9]|9[0-9])\d{8}')
    while True:
        s = re.search(t, text)
        if s:
            text = text.replace(s.group(), '***********')
        else:
            break
    return {"output_text": text}


transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=anonymize
)

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)
prompt = PromptTemplate(
    input_variables=["input"],
    template="根据下述句子，提取候选人的职业:\n{input}\n输出JSON, 以job为key",
)

task_chain = LLMChain(llm=llm, prompt=prompt)

overall_chain = SimpleSequentialChain(
    chains=[transform_chain, task_chain], verbose=True)

print(overall_chain.run("我是程序员，有事随时跟我联系，打我手机13912345678"))


# 常用chain类型之Router
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chains.router import MultiPromptChain
import warnings

warnings.filterwarnings("ignore")

windows_template = """
你只会写DOS或Windows Shell脚本。你不会写任何其他语言的程序。你也不会写Linux脚本。

用户问题:
{input}
"""

linux_template = """
你只会写Linux Shell脚本。你不会写任何其他语言的程序。你也不会写Windows脚本。

用户问题:
{input}
"""

prompt_infos = [
    {
        "name": "WindowsExpert",
        "description": "擅长回答Windows Shell相关问题",
        "prompt_template": windows_template,
    },
    {
        "name": "LinuxExpert",
        "description": "擅长回答Linux Shell相关问题",
        "prompt_template": linux_template,
    },
]

llm = OpenAI()

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain, # default chain when no destination chain matched
    verbose=True,
)

print(chain.run("帮我写个脚本，让Windows系统每天0点自动校对时间"))

print(chain.run("帮我写个cron脚本，让系统每天0点自动重启"))

print(chain.run("今天天气不错啊"))

# 调用 OpenAI Function Calling 获得 Pydantic 输出
from pydantic import BaseModel, Field
from typing import Optional
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


class Contact(BaseModel):
    """抽取联系人信息"""

    name: str = Field(..., description="联系人姓名")
    address: str = Field(..., description="联系人地址")
    tel: str = Field(None, description="联系人电话")


prompt_msgs = [
    SystemMessage(
        content="你是信息抄录员。"
    ),
    HumanMessage(
        content="根据给定个数从下面的句子中抽取信息:"
    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    HumanMessage(content="Tips: Make sure to answer in the correct format"),
]
prompt = ChatPromptTemplate(messages=prompt_msgs)
llm = ChatOpenAI(model="gpt-4-0613", temperature=0)

chain = create_openai_fn_chain([Contact], llm, prompt, verbose=True)

chain.run("寄给亮马桥外交办公大楼的王卓然，13012345678")


# 基于Document Chain的不同实现思路
import wordninja
from langchain.callbacks import StdOutCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.document_loaders import PyPDFLoader

# 把整个chain的verbose打开
def set_verbose_recusively(chain):
    chain.verbose = True
    for attr in dir(chain):
        if attr.endswith('_chain') and isinstance(getattr(chain, attr), Chain):
            subchain = getattr(chain, attr)
            set_verbose_recusively(subchain)


loader = PyPDFLoader("ftMLDE-2021.pdf")
documents = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400,
    length_function=len,
    add_start_index=True,
)
def preprocess(text):
    def split(line):
        tokens = re.findall(r'\w+|[.,!?;%$-+=@#*/]', line)
        return [
            ' '.join(wordninja.split(token)) if token.isalnum() else token
            for token in tokens
        ]

    lines = text.split('\n')
    for i,line in enumerate(lines):
        if len(max(line.split(' '), key = len)) >= 20:
            lines[i] = ' '.join(split(line))
    return ' '.join(lines)

paragraphs = text_splitter.create_documents(
    [preprocess(d.page_content) for d in documents[2:20]])
# print(paragraphs)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
db = FAISS.from_documents(paragraphs, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff", # can be refine, map_reduce, map_rerank
    retriever=db.as_retriever(),
    verbose=True
)
set_verbose_recusively(qa_chain)

query = "how to build MLDE model?"
qa_chain.run(query)



import configparser
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.schema.runnable import RunnableLambda


# 定义矩阵展开函数
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


config = configparser.ConfigParser()
config.read('../../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

content = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()[0].page_content

splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)

prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in splitter.split_text(x)]
)

# trunks = prep.invoke(content)


# 创建paper类
class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


# 创建Info类
class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]


funcs = [
    convert_pydantic_to_openai_function(Info)
]

model = ChatOpenAI(openai_api_key=openai_api_key).bind(functions=funcs, function_call={"name": "Info"})

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 
Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.
Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

extraction_chain = prompt | model | JsonKeyOutputFunctionsParser(key_name="papers")

# extraction_chain.invoke({"input": content})

chain = prep | extraction_chain.map() | flatten
response = chain.invoke(content)

print(response)

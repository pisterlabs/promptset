import configparser
from langchain.document_loaders import WebBaseLoader
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser
from typing import Optional, List

config = configparser.ConfigParser()
config.read('../../config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')


def tagging_chain_web():
    doc = WebBaseLoader(web_path="https://tech.ifeng.com/c/8VEctgVlwbk").load()[0]
    content = doc.page_content
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-1106")

    class Overview(BaseModel):
        """文本内容的摘要信息"""
        summary: str = Field(description="一段文本的总结概括")
        language: str = Field(description="文本使用的语言")
        keywords: str = Field(description="这段文本中的关键字")

    funcs = [
        convert_pydantic_to_openai_function(Overview)
    ]
    llm = llm.bind(functions=funcs, function_call={"name": "Overview"})
    # 创建prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "抽取相关信息，如果没有提供的话不要瞎猜，只抽取已提供的信息"),
        ("human", "{input}")
    ])
    tagging_chain = prompt | llm | JsonOutputFunctionsParser()
    response = tagging_chain.invoke({"input": content})
    print(response)


# tagging_chain_web()

def extraction_chain_web():
    doc = WebBaseLoader(web_path="https://tech.ifeng.com/c/8VEctgVlwbk").load()[0]
    content = doc.page_content[:3000]
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

    class News(BaseModel):
        """Information about news mentioned."""
        title: str
        author: Optional[str]

    class Info(BaseModel):
        """Information to extract"""
        news: List[News]

    funcs = [convert_pydantic_to_openai_function(Info)]
    llm = llm.bind(functions=funcs, function_call={"name": "Info"})
    # 创建prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
        ("human", "{input}")
    ])
    chain = prompt | llm | JsonKeyOutputFunctionsParser(key_name="news")
    response = chain.invoke({"input": content})
    print(response)


extraction_chain_web()

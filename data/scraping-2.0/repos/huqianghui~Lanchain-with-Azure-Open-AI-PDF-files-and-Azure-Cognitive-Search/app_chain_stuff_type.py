from openAIRoundRobin import get_openaiByRoundRobinMode
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQAWithSourcesChain
from azureCognitiveSearch import AzureCognitiveSearch6Retriever,AzureCognitiveSearchVectorRetriever,AzureCognitiveSearchHybirdRetriever,AzureCognitiveSearchSenamicHybirdRetriever
from dotenv import load_dotenv
import os
import json
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path)

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.
Here's your output format:
{{
  "人口": "14",
  "单位": "亿"
}}

In the json result,only display the attributes and units to be queried, or only display the queried attributes. Do not display any other  attribute in the json result. Do not display any other attribute in the json result. Do not display any other attribute in the json result.

SOURCES:

QUESTION: {question}
=========
{summaries}
=========
ANSWER:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qaSenamicHybirdChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=AzureCognitiveSearchSenamicHybirdRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

# result = qaChain("销售金额:南方区域")
# print("AzureCognitiveSearchSenamicHybirdRetriever: ")
# print(result)

qaVectorHybirdChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=AzureCognitiveSearchHybirdRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

qaVectorChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=AzureCognitiveSearchVectorRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

qaSearchChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=AzureCognitiveSearch6Retriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

def is_json_string(json_string):
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False

def get_content_with_multi_method(index_content):
    result = qaSenamicHybirdChain(index_content)
    if(is_json_string(result["answer"])):
        return result
    else:
        print(">> try qaVectorHybirdChain......")
        result = qaVectorHybirdChain(index_content)
        if(is_json_string(result["answer"])):
            return result
        else:
            print(">> try qaVectorChain......")
            result = qaVectorChain(index_content)
            if(is_json_string(result["answer"])):
                return result
            else:
                print(">> try qaSearchChain......")
                result = qaSearchChain(index_content)
                return result

def get_content_by_index_content(index_content):
    try:
        result = get_content_with_multi_method(index_content)
        print(result["answer"])
        return result["answer"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

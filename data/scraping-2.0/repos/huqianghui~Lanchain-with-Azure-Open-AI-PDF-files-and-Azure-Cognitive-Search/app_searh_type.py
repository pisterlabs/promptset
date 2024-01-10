from openAIRoundRobin import get_openaiByRoundRobinMode
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQAWithSourcesChain
from azureCognitiveSearch import AzureCognitiveSearchVectorRetriever,AzureCognitiveSearchHybirdRetriever,AzureCognitiveSearchSenamicHybirdRetriever
from dotenv import load_dotenv
import os
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
# ## search type: vector
# qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
#                                          chain_type="stuff", 
#                                          retriever=AzureCognitiveSearchVectorRetriever(), 
#                                          verbose=True,
#                                          chain_type_kwargs=chain_type_kwargs)

# result = qaChain("销售金额:南方区域")
# print("AzureCognitiveSearchVectorRetriever: ")
# print(result)


# ## search type: hybird(word + vector)
# qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
#                                          chain_type="stuff", 
#                                          retriever=AzureCognitiveSearchHybirdRetriever(), 
#                                          verbose=True,
#                                          chain_type_kwargs=chain_type_kwargs)

# result = qaChain("销售金额:南方区域")
# print("AzureCognitiveSearchHybirdRetriever: ")
# print(result)


# search type: senamic hybird(senamic + vector)
qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="stuff", 
                                         retriever=AzureCognitiveSearchSenamicHybirdRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

result = qaChain("合同金额:已售资源未竣工")
print("AzureCognitiveSearchSenamicHybirdRetriever: ")
print(result)
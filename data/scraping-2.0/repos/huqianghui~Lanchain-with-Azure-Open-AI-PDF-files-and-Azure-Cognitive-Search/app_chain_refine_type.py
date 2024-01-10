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

question_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.
Here's your output format:
{{
  "人口": "14",
  "单位": "亿"
}}

The context below.\n
------------\n
{context_str}\n
------------\n

answer the question:: {question} \n
"""

QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["question", "context_str"]
)


refine_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
And You are an assistant designed to extract data from text. Users will paste in a string of text and you will respond with data you've extracted from the text as a JSON object.
Here's your output format:
{{
  "人口": "14",
  "单位": "亿"
}}

The original question is as follows: {question} \n

We have provided an existing answer, including sources: {existing_answer}\n
We have the opportunity to refine the existing answer
(only if needed) with some more context below.\n
------------\n
{context_str}\n
------------\n

Given the new context, refine the original answer to better 
answer the question. 
If you do update it, please update the sources as well.
If the context isn't useful, return the original answer.
"""

REFINE_PROMPT = PromptTemplate(
    template=refine_prompt_template, input_variables=["question", "existing_answer", "context_str"]
)
chain_type_kwargs = {"question_prompt":QUESTION_PROMPT,"refine_prompt": REFINE_PROMPT}
qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="refine", 
                                         retriever=AzureCognitiveSearchSenamicHybirdRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

result = qaChain("销售金额:南方区域")
print("AzureCognitiveSearchSenamicHybirdRetriever: ")
print(result)
# step 3) insert result into excel column, if result is valid number


#### test
#getResult("销售金额:南方区域")
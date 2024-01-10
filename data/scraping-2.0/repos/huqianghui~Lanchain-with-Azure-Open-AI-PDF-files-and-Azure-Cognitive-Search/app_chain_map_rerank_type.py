from openAIRoundRobin import get_openaiByRoundRobinMode
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
from azureCognitiveSearch import AzureCognitiveSearchVectorRetriever,AzureCognitiveSearchHybirdRetriever,AzureCognitiveSearchSenamicHybirdRetriever
from dotenv import load_dotenv
import os
from langchain.output_parsers.regex import RegexParser

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

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

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

The context below.\n
------------\n
{context}\n
------------\n

answer the question:: {question} \n
Helpful Answer:
"""

QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, 
    input_variables=["question", "context"],
    output_parser=output_parser
)

chain_type_kwargs = {"prompt":QUESTION_PROMPT}
qaChain = RetrievalQAWithSourcesChain.from_chain_type(llm=get_openaiByRoundRobinMode(), 
                                         chain_type="map_rerank", 
                                         retriever=AzureCognitiveSearchSenamicHybirdRetriever(), 
                                         verbose=True,
                                         chain_type_kwargs=chain_type_kwargs)

result = qaChain("销售金额:南方区域")
print("AzureCognitiveSearchSenamicHybirdRetriever: ")
print(result)
# step 3) insert result into excel column, if result is valid number


#### test
#getResult("销售金额:南方区域")
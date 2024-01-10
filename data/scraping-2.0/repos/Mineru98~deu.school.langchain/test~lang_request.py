# -*- coding:utf-8 -*-
import os, sys
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMRequestsChain


try:
    load_dotenv()
    apikey = os.environ["OPENAI_API_KEY"]
except:
    print("Please, export your OpenAI API KEY over 'OPENAI_API_KEY' environment variable")
    print("You may create the key here: https://platform.openai.com/account/api-keys")
    sys.exit(1)


template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""

llm = OpenAI(temperature=0, model_name="text-davinci-001")
PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))
# question = "What are the Three (3) biggest countries, and their respective sizes?"
# 동의대학교 홈페이지는?
question = input("질문) ")
inputs = {"query": question, "url": "https://www.google.com/search?q=" + question.replace(" ", "+")}
res = chain(inputs)
output = res["output"].strip()
if output != "not found":
    print(output)
else:
    print("결과 없음")

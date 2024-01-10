#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.utilities import PythonREPL
from langchain import LLMMathChain

'''
chatgpt在数学运算方面，水平很差，需要外部能力
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# 直接做数学运算，效果很差
def get_math(llm, question):
    multiply_prompt = PromptTemplate(template="请计算一下{question}是多少?", input_variables=["question"])
    math_chain = LLMChain(llm=llm, prompt=multiply_prompt, output_key="answer")
    answer = math_chain.run({"question": question})
    print("OpenAI API 说答案是:", answer)


# 那就先生成代码，然后运行代码得到结果
def get_math_by_python(llm, question): 
    multiply_by_python_prompt = PromptTemplate(template="请写一段Python代码，计算{question}?", input_variables=["question"])
    math_chain = LLMChain(llm=llm, prompt=multiply_by_python_prompt, output_key="answer")
    answer_code = math_chain.run({"question": question})

    python_repl = PythonREPL()
    result = python_repl.run(answer_code)
    print("OpenAI API 先生成:", result)


# 也可以使用LLMMathChain达到相同的目的
def get_math_by_math_chain(llm, question): 
    llm_math = LLMMathChain(llm=llm, verbose=True)
    result = llm_math.run(question)
    print(result)


if __name__ == '__main__':
    get_api_key()

    python_answer = 352 * 493
    print("Python 说答案是:", python_answer)

    llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)
    get_math(llm, "352乘以493")
    get_math_by_python(llm, "352乘以493")
    get_math_by_math_chain("请计算一下352乘以493是多少?")

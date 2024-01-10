from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
import time

question_template = PromptTemplate(
    input_variables=['topic'],
    template='Please generate sustainable environment related question for kids related to {topic} that requires a 1-word answer.'
)

correct_template = PromptTemplate(
    input_variables=['question'],
    template='In less than 3 words answer to {question}'
)

wrong_template = PromptTemplate(
    input_variables=['question'],
    template='In less than 3 words give wrong answer to {question}'
)

llm = OpenAI(temperature=0.9)

question_chain = LLMChain(llm=llm, prompt=question_template, output_key='question')
correct_chain = LLMChain(llm=llm, prompt=correct_template, output_key='correct')
wrong_chain = LLMChain(llm=llm, prompt=wrong_template, output_key='wrong')

def answers(correct, wrong):
    num = random.randint(0, 1)
    list = []
    if num == 0:
        list.append(correct)
        list.append(wrong)
    else:
        list.append(wrong)
        list.append(correct)
    return list, num

def cur(num, cur_num):
    if num == cur_num:
        return True
    return False

def quiz(prompt):
    question = question_chain.run(prompt)
    correct = correct_chain(question)
    wrong = wrong_chain(question)
    ans, num = answers(correct.get("correct"), wrong.get("wrong"))
    return question,ans








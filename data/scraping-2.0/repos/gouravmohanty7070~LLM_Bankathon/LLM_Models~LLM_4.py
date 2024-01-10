from dotenv import load_dotenv, find_dotenv
import openai
import os
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


data=''
with open(r"test\transcript.txt", 'r') as transcript:
    lines=transcript.readlines()
    data="\n".join(lines)

chat = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k")

jd="Mechanical Engineer intern"

system_prompt = f"""
You are an AI model that scores Answers given by candidates to a a job interview. 
Both question and it's answer given by interviewee will be provided to you in the form of a transcript. 
There may be some words spelled wrong or joined words or some other formatting issue as they been converted from speech to text, try and fix them yourself. 

The most important scoring criteria is relevance the to the job description and correctness of the answer
The Job description will be given to you in triple back ticks
```{jd}```
Maximum score of each answer can be 20
return questions and its original answer and a number with the score of the answer. 
give the total sum of the score in the end

Output should be like: 
    Q1 <starting20 characters of question> -> 15
    (answer of the candidate as it was)
    Q2 <starting20 characters of question> -> 13
    (answer of the candidate as it was)
    Q3 <starting20 characters of question> -> 12
    (answer of the candidate as it was)
    .
    .
    .

    sum : 200

DO not score greetings like hello, Thank you, welcome etc as they do not come under questions and do not mention them as questions in the final output
DO not overscore or underscore. Keep it realistic. 
DO NOT alter or summarize the question while providing the output. Keep it as it was. 
IMPORTANT do not reply with "As an AI model..." under any circumstances 
"""


def func_(data):
    store = chat(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=data)
        ]
    )
    return store

store = func_(data=data)
print(store.content)

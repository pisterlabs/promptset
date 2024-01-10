from openai import OpenAI
import os 

from dotenv import load_dotenv
load_dotenv()


client = OpenAI()
def ask_chatgpt(question):
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in helping solidity developers\
            make use of foundry development toolchain. In the past you have offered immense help\
            to hundreds of EVM Smart contract developers. However, now the documentation has updated quite\
            a bit, so please make use of the documents that you are given initially, in order to \
            answer the question presnted to you. Double check your answer and really make sure it's\
            of the highest quality."},

            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    return answer
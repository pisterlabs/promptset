from openai import OpenAI
import os 

from dotenv import load_dotenv
load_dotenv()


client = OpenAI()
def ask_chatgpt(question):
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in fixing solidity security vulnerabilities. You have given code\
            suggestion in the past that have tremendously improved the lives of hundreds of\
            security researchers and blockchain developers on EVM based chains."},

            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    return answer
        

fileopen = open("Data\\Api.txt","r")
API = fileopen.read()
fileopen.close()
# print(API)

#importing
import openai
import time
from dotenv import load_dotenv

#coding
openai.api_key = API
load_dotenv()
completion = openai.Completion()

def ReplyBrain(question,chat_log = None):
    # Filelog = open("DataBase\\chat_log.txt","r")
    Filelog = open("DataBase\\qna_log.txt","r")
    chat_log_template = Filelog.read()
    Filelog.close()
    if chat_log is None:
        chat_log = chat_log_template
    prompt = f'{chat_log}You: {question}\nJarvis: '
    response = completion.create(
        model = "text-davinci-002", 
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 60,
        top_p = 0.3,
        frequency_penalty=0.5,
        presence_penalty = 0)
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + f"\nYou : {question} \nJarvis : {answer}"
    Filelog = open("DataBase\\qna_log.txt","w")
    Filelog.write(chat_log_template_update)
    Filelog.close()
    return answer





























# while True: 
#     kk = input("Enter: ")
#     print(ReplyBrain(kk))



# import os
# # import openai
# from config import apiKey

# # print(apiKey)

# # OpenAI.api_key = apiKey


# This code is for v1 of the openai package: pypi.org/project/openai
# from openai import OpenAI
# import openai

# # Replace "your-api-key" with your actual OpenAI GPT-3 API key
# openai.api_key = "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"


# try:
#     # Your API request code here
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "write an email to my boss for resignation?"},
#             {"role": "user", "content": "write an email to my boss for resignation?"}
#         ],
#         temperature=1,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
# except openai.error.RateLimitError as e:
#     print(f"Rate limit exceeded: {e}")
#     # Implement logic to handle rate-limiting, such as waiting and retrying after a delay
# except openai.error.OpenAIError as e:
#     print(f"OpenAI API error: {e}")
#     # Handle other OpenAI API errors

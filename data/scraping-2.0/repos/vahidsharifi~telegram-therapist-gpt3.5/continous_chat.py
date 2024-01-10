# from dotenv import load_dotenv
# from random import choice
# import openai
# import os
# from flask import Flask, request

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# completion = openai.Completion()





# prompt_text = [{"role": "system", "content": "You are a funny casino assisstant with taste of humor. Your name is Siroos. You just answer the questions related to casino."}]
# # Creating the main gpt-interactive function
# def ask(question, chat_log=None):
#     global prompt_text
#     prompt_text.append({"role": "user", "content": f"{question}"})
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=prompt_text,
#         temperature=0.9,
#         max_tokens=150,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0.6,
#         stop=["\n"]
#     )
#     story = response['choices'][0]['message']['content']
#     if len(prompt_text) == 6 :
#         prompt_text[1] = prompt_text[3]
#         prompt_text[2] = prompt_text[4]
#         prompt_text[3] = prompt_text[5]
#         prompt_text = prompt_text[0:4]
#         prompt_text.append({"role": "assistant", "content": str(story)})
#     else:
#         prompt_text.append({"role": "assistant", "content": str(story)})
#     return str(story)
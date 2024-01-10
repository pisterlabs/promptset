import openai
import os
from dotenv import load_dotenv
from datetime import datetime, time

zero_time = datetime.combine(datetime.today(), time(0,0))
print(zero_time)
# def function_for_start():
#     try:
#         text = '''Напиши 10 названий бренда на русском и 10 других вариантов на английском основанных на ключевых словах: строгость, элегантность
#         женственность
#         подчеркнутая чувственность, вдохновение и изысканность
#         Наталия, минимализм, эстетика, nut, nut’s'''
#         new = []
#         new.append(
#             {'role': 'user', 'content': text})
#         completion = openai.ChatCompletion.create(
#             model='gpt-4',
#             messages=new,
#             temperature=0
#         )
#         chat_gpt_response = completion.choices[0].message.content
#         print(chat_gpt_response)
#     except Exception as e:
#         print(e)

# function_for_start()


# # keyword = 'LINK'
# # chat_gpt_response = '''С удовольствием, вот ваш текст: LINK https://world.com

# # А теперь давайте вернемся к нашему обсуждению. Расскажите, были ли клиенты, которые пришли к вам из вашей группы ВКонтакте?'''
# # if keyword in chat_gpt_response:
# #     index = chat_gpt_response.index(keyword) + len(keyword) + 1
# #     vk_link = chat_gpt_response[index:].split()[0]
# #     print(vk_link)

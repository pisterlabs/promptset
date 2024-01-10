'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 
'''


import os
from langchain.llms import Aviary

llm = Aviary(model='amazon/LightGPT', aviary_url=os.environ['AVIARY_URL'], aviary_token=os.environ['AVIARY_TOKEN'])

result = llm.predict('What is the meaning of love?')
print(result)
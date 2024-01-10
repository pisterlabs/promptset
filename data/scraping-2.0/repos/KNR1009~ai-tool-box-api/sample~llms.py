from langchain.llms import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# 利用したいモデル
model_name = "gpt-3.5-turbo" 
# チャットモデルを利用する
llm = OpenAI(openai_api_key=openai_api_key, temperature=0, model_name=model_name)

# 実行
print(llm("ITエンジニアついて30文字で教えて"))
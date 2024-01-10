from langchain.llms import OpenAI
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("输入apiKey: ")

llm = OpenAI()
response = llm.predict("我想要新建一个自媒体视频账号，用于记录自己的健身日常，请帮我取一个好名字，要求是中文")
print(response)
from langchain.llms import OpenAI
import os
import getpass

# 方法1：硬编码
open_ai_key = "xxx"
llm = OpenAI(openai_api_key=open_ai_key)
# 或者通过硬编码设置环境变量, langchain会自动读取
os.environ["OPENAI_API_KEY"] = "xxxxx"
llm = OpenAI()

# 方法2：在操作系统中设置环境变量
'export OPENAI_API_KEY="xxxxxx"'

# 方法3: 使用getpass设置环境变量
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAi Api key: ")

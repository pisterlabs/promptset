from revChatGPT.V1 import Chatbot
import yaml
import openai

# 读取YAML文件
with open('script/translate_file/config.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# 读取配置
access_token_list = data['access_token_list']
access_token = access_token_list[0]
BASE_URL = data['BASE_URL']

# openai.api_key = "这里填 access token，不是 api key"
openai.api_base = BASE_URL
openai.api_key = access_token

chatbot = Chatbot(config={
  "access_token": access_token
})

# print("Chatbot: ")
# prev_text = ""
# for data in chatbot.ask(
#     "Hello world",
# ):
#     message = data["message"][len(prev_text) :]
#     print(message, end="", flush=True)
#     prev_text = data["message"]
# print()

# prompt = "how many beaches does portugal have?"
# response = ""
# for data in chatbot.ask(
#   prompt
# ):
#     response = data["message"]
# print(response)

# 定义问题
question = "如何系统全面的学习ChatGPT知识，请推进一些学习资源"

# 输出问题
print(question)

# 输出ChatGPT的回答
print("ChatGPTBot: ")

# 定义prev_text变量，用于保存上一次对话的文本内容
prev_text = ""

# 通过ask方法向ChatGPT发送问题并获取回答
for data in chatbot.ask(question):

    # 从回答数据中提取ChatGPT的回答，并去除前面已经输出过的文本部分
    message = data["message"][len(prev_text) :]

    # 输出ChatGPT的回答
    print(message, end="", flush=True)

    # 更新prev_text变量
    prev_text = data["message"]

# 输出空行，以便下一轮对话
print()


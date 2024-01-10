import json
import os
import openai

# 构建config.json文件的路径
config_file_path = os.path.join("config", "config.json")

# 读取配置文件
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

# 设置环境变量
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

#获取api_key函数
def get_api_key(key):
    if key == '':
        openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.api_key = key

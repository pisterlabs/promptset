# 这个是第一个例子，用来测试是否能够连接到 OpenAI 服务
# 在运行之前，需要先安装 openai 包，可以使用 pip install openai 命令来安装

# 请确保你已经在 .env 文件中配置了 OPENAI_API_KEY 和 OPENAI_API_BASE 两个环境变量
## .env文件就是放在与本脚本同一个目录中，没有名字，后缀为.env的文件
## 文件内容为：
## OPENAI_API_KEY="你的API_KEY"（格式为双引号包裹的一个key，例如"sk-TufGVICcOLQzTb2awerfeSeGJq2rf3ae53Rcz4eg4JPlQ"）
## OPENAI_API_BASE="你的API_BASE"（格式为双引号包裹的一个url，例如"https://api.openai.com"，"https://api.fe8.cn/v1"）
## 请注意，这两个环境变量的值都是需要加双引号的，否则会报错
## 如果拿不准API_BASE的值是否正确，可以在浏览器中拷贝输入此此地址，比如"https://api.openai.com"，会看到如下信息：
#{
#    "message": "Welcome to the OpenAI API! Documentation is available at https://platform.openai.com/docs/api-reference"
#}

# 如果用的是VSCode，点右上角三角，运行此脚本，如果在屏幕下方输出中看到以下信息，就说明连接成功了：
# ChatCompletion(id='chatcmpl-...', choices=...)
# 如果看到以下数字和信息，说明连接失败了：
# 404 Not Found， ERR_CONNECTION_TIMED_OUT（网址有错误，找不到；或无法访问（需要科学上网））
# 401 Authentication Failed， ERR_INVALID_AUTH_HEADER，Invalid API key（API_KEY有错误）
# 其他错误信息，可用ChatGpt、文心一言问一下。

import os
from openai import OpenAI

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 配置 OpenAI 服务

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

print(response)
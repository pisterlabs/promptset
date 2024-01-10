from openai import OpenAI
import os
import json

os.environ["http_proxy"] = "http://127.0.0.1:7899"
os.environ["https_proxy"] = "http://127.0.0.1:7899"

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

class BaseChat:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.history = [
            {
                "role": "system",
                "content": "你是一个专业的命令行助手，擅长用命令行解决各种问题。每次回答问题时，你都会以json格式给出linux系统和windows系统各一个命令行的解决方案，而且不说其他任何话。当被要求创建文件时，请自动选择合适的文件名以及编码。不允许删除或改动的文件是[base_chat.py, path_structure.py, gradio_interface.py, cli_helper.py]!!"
            },
            {
                "role": "user",
                "content": "我想知道当前目录下有哪些文件"
            },
            {
                "role": "assistant",
                "content": "{\"linux\": \"ls\", \"windows\": \"dir\"}"
            },
            {
                "role": "user",
                "content": "以utf-8编码保存一个画sinx的python代码，放在example_dir下"
            },
            {
                "role": "user",
                "content": '{"linux": "echo -e \"import math\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 2 * np.pi, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.show()\" > example_dir/sinx.py", "windows": "echo import math> example_dir\\sinx.py && echo import matplotlib.pyplot as plt >> example_dir\\sinx.py && echo import numpy as np >> example_dir\\sinx.py && echo x = np.linspace(0, 2 * np.pi, 100) >> example_dir\\sinx.py && echo y = np.sin(x) >> example_dir\\sinx.py && echo plt.plot(x, y) >> example_dir\\sinx.py && echo plt.show() >> example_dir\\sinx.py"}'
            }
        ]

    def chat(self, message):
        wrap_message = {
            "role": "user",
            "content": message
        }
        self.history.append(wrap_message)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=self.history,
            response_format={"type": "json_object"}
        )
        self.history.append(
            {
                "role": "assistant",
                "content": json.loads(completion.choices[0].json())["message"]['content']
            }
        )
        return self.history[-1]["content"]


if __name__ == "__main__":
    chat = BaseChat()
    while True:
        message = input(">>> ")
        print(chat.chat(message))

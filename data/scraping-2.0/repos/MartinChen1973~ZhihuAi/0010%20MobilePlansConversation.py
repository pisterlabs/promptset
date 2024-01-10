# 这个例子演示如何编码实现与AI反复对话

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

# 基于 prompt 生成文本
## 以下是一个简单的函数（或者叫“方法”），用来调用大模型生成文本
def get_completion(messages, model="gpt-3.5-turbo"):      # 默认使用 gpt-3.5-turbo 模型
    messages = messages # 此处的 messages 是一个列表，列表中的每个元素都是一个字典，包含两个键值对：role 和 content
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,                                  # 模型输出的随机性，0 表示随机性最小
    )
    if response is not None:
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content          # 返回模型生成的文本
            # 处理生成的文本
        else:
            print("No choices in response")
            print(response)
    else:
        print("No response received")

# 任务描述
instruction = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称，月费价格，月流量。
当前只有以下4个流量套餐产品，分别是：
经济套餐，月费50元，10G流量；
畅游套餐，月费180元，100G流量；
无限套餐，月费300元，1000G流量；
校园套餐，月费150元，200G流量，仅限在校生。
根据用户输入，并为其选择一个套餐，并用人类的语言向其推荐。
"""

# 输出格式
output_format = """
以 JSON 格式输出
"""

backGroundPrompt = f"""
{instruction}

{output_format}
"""

messages = [{"role": "system", "content": backGroundPrompt}]

print(f"=小瓜=：我是套餐销售员小瓜，请问有什么可以帮助您的？")
# 循环10次，每次都调用大模型生成文本

for i in range(10):
    # 用户输入
    input_text = input("用户输入：")
    if (input_text.__contains__("再见")):
        break
    messages.append({"role": "user", "content": input_text})

    # 调用大模型生成文本
    response = get_completion(messages)

    # 将生成的文本作为系统输出
    messages.append({"role": "assistant", "content": response})

    #打印对话中assistant的最后一句
    print(f"=小瓜=：{messages[-1].get('content')}")


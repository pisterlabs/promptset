# 这个例子演示如何用提示词给一个用户购买手机套餐的对话进行补全
# 在运行之前，请确保上一课程中的Api已经正确配置

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
def get_completion(prompt, model="gpt-3.5-turbo"):      # 默认使用 gpt-3.5-turbo 模型
    messages = [{"role": "user", "content": prompt}]    # 将 prompt 作为用户输入
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
根据用户输入，识别用户在上述三种属性上的倾向，并为其选择一个套餐。
"""

# 用户输入
input_text = """
我是学生，想要一个流量大还便宜的套餐。流量要大于50G。
"""
input_text = """
办个100G的套餐。
"""

# 输出格式
output_format = """
以 JSON 格式输出
"""

# 稍微调整下咒语，加入输出格式
prompt = f"""
{instruction}

{output_format}

用户输入：
{input_text}
"""

# 调用大模型
response = get_completion(prompt)

print(response)
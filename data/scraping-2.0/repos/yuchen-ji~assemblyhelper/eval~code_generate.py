import os
import openai

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

openai.api_key = os.getenv(
    "OPENAI_KEY", default="sk-GM3AyFSCFHwbJdnC4c1a2637E4Bf4433AcFcAc8c3e976cFe"
)
openai.api_base = "https://api.ai-yyds.com/v1"


# load prompt message
file_path = "eval/prompts/robot_prompt_update8.yml"  # 替换为您的文件路径
with open(file_path, "r", encoding="utf-8") as file:
    prompt = file.read()

messages = [
    {
        "role": "system",
        "content": "You are a desktop robotic arm with 6 degrees of freedom, and the end effector is a gripper. You need to understand my actions/language and assist me in completing the assembly of the parts.",
    },
    {
        "role": "user",
        "content": prompt,
    },
]

while True:
    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        # model="gpt-3.5-turbo",
        messages=messages,  # prompt
        temperature=0.2,  # 0~2, 数字越大越有想象空间, 越小答案越确定
        n=1,  # 生成的结果数
        # top_p=0.1, # 结果采样策略，0.1只采样前10%可能性的结果
        # presence_penalty=0,  # 主题的重复度 default 0, between -2 and 2. 控制围绕主题程度，越大越可能谈论新主题。
        # frequency_penalty=0,  # 重复度惩罚因子 default 0, between -2 and 2. 减少重复生成的字。
        # stream=False,
        # logprobs=1,  # Modify the likelihood of specified tokens appearing in the completion.
        # stop="\n"  # 结束字符标记
    )

    chat_response = completion
    answer = chat_response.choices[0].message.content
    print(f"ChatGPT: {answer}")
    messages.append({"role": "assistant", "content": answer})

    # 用户输入新的请求
    content = ""
    str = input("User: ")
    while str != "q":
        content = content + str + "\n"
        str = input("User: ")
    # print(content)

    messages.append({"role": "user", "content": content})

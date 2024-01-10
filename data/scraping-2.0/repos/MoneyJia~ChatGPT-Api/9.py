import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

contextMessages = [
    # GPT角色设定
    {"role": "system", "content": "你是一个资深的心理咨询师"},
    # 模拟用户输入信息
    {"role": "user", "content": "我觉得GPT很酷！"}
]


def run():
    openai.api_key = "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"

    chat_completion = openai.ChatCompletion.create(
        # 选择的GPT模型
        model="gpt-3.5-turbo-16k-0613",
        # 上下文
        messages=contextMessages,
        # 1.2使得GPT答复更具随机性
        temperature=1.2,
        # 不采用流式输出
        stream=False,
        # 期望GPT每次答复两条（这里只是为了演示，正常情况取值为1）
        n=2,
    )

    print(chat_completion)
    
    # 第一个答复
    print(chat_completion.choices[0].message.content)
    # 第二个答复，上边的n>=2时，才会有该条回复
    print(chat_completion.choices[1].message.content)

    # 将答复存储到上下文中，否则下次再进行对话时，GPT会遗忘之前的答复
    contextMessages.append(chat_completion.choices[0].message)

run()
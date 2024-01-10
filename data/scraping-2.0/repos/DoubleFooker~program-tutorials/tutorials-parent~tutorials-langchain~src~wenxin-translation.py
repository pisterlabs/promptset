from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_wenxin import ChatWenxin

WENXIN_APP_Key = "IwADtq6gmtSo2TNuAoLYwAGw"
WENXIN_APP_SECRET = "cbftIc2ntYUYDO4qK9FEKgydsl1nDryM"


def translate(input_text):
    # 模板
    template = """你是一个把英文翻译成中文的助手，请把下面input标签内的内容翻译成中文，如果内容只有一个单词或短语，请给出音标，中文和例句；如果内容是句子，请只给出中文。

<input>{text}</input>
翻译结果：
"""

    # 使用template创建HumanMessagePromptTemplate，模板中含有一个text变量
    human_message_prompt = HumanMessagePromptTemplate.from_template(template)

    # 创建ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    # 使用传入的input_text参数对text进行赋值，创建messages
    messages = chat_prompt.format_prompt(text=input_text).to_messages()
    #print('input content: \n', messages[0].content)

    chat_model = ChatWenxin(
        temperature=0.9,
        model="ernie-bot-turbo",
        baidu_api_key = WENXIN_APP_Key,
        baidu_secret_key = WENXIN_APP_SECRET,
        verbose=True,
    )

    # 调用文心API
    response = chat_model(
        messages
    )
    # 返回结果
    return response


if __name__ == "__main__":
    input_text = input("请输入需要翻译的英文:")
    # 无输入退出
    while input_text:
        response = translate(input_text)
        print("翻译结果: \n", response.content)
        input_text = input("请输入需要翻译的英文:")

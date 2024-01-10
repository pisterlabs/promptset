from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage, SystemMessage

if __name__ == "__main__":

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("あなたは{country}料理のプロフェッショナルです。"),
        HumanMessagePromptTemplate.from_template("以下の料理のレシピを考えてください。\n\n料理名：{dish}")
    ])

    messages = chat_prompt.format_prompt(country="イギリス", dish="肉じゃが").to_messages()

    print(messages)
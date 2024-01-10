import time

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model_name="gpt-4")

template="あなたはお笑い芸人です．ユーザーに渡される言葉がダジャレかどうか判定してください．ダジャレだった場合は`True`、そうでない場合は`False`と答えてください．理解できない場合も`False`と答えてください．"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


def is_boke(boke: str) -> bool:
    start = time.time()
    # print("---------------------------------------------------------------")
    # print("boke: ", boke)
    # chain = prompt | llm

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    output = chat(chat_prompt.format_prompt(text=boke).to_messages())

    # output = chain.invoke({"boke": boke})

    # print("output: ", output.content)
    # print("output type: ", type(output))

    end = time.time()
    print("gag_judegement time: " + str(end - start))

    return output.content == "True"

if __name__ == "__main__":
    boke1 = "布団が吹っ飛んだ"
    boke2 = "今日は暑い日だ"

    print(is_boke(boke1))
    print(is_boke(boke2))
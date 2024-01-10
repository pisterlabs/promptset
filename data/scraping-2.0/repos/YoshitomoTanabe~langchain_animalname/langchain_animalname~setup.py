import os
import dotenv
import sys
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from . import conversation


def choice():
    print(
        "実行するlangchainを選択してください(llm/chat): ",
        end="",
    )

    choiced_number = input()
    if choiced_number != "llm" and choiced_number != "chat":
        print("無効な選択です")
        sys.exit()
    return choiced_number


def llm():
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(temperature=1.0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["cute_object"], template="{cute_object}の可愛い名前を5つ日本語で出力せよ。"
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    conversation.user_instruct(
        "好きな動物を入力してください。かわいい名前をかんがえるよ。\n動物の種類を入力(「おわり」で終了)", chain
    )


def chat():
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    template = """
            あなたは物事に命名をするアシスタントです。指示された印象を与える名前を5つ考えてください。
            印象とは物事に対する形容詞です。例えば「かわいい」、「こわい」、「かっこいい」などが挙げられます。
            もし指示内に印象に関する指定が含まれなかった場合は、
            「印象に関しては特に指定がなかったため、かわいい名前を考えました。」と断りを入れた上で、かわいい名前を提示してください。
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "{instruction}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True),
        prompt=chat_prompt,
    )

    try:
        print(chain.run(sys.argv[1]))
    except IndexError:
        print("コマンドライン引数に指示を渡してください。(例: 犬の可愛い名前を考えてください。)")

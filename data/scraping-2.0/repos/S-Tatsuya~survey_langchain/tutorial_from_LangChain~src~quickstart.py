import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser, HumanMessage


def tutorial_of_llms():
    """LLM, Chat Modelのpredict, predict_messagesの動きを確認する
    1. 'こんにちは' を `predict` に与える
    2. 'text'を変数に格納して `predict` に与える
    3. '配列'を `predict_messages` に与える"""
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    chat_model = ChatOpenAI()

    print("\033[41mLLMs First Prompts: predict\033[0m")
    print("----- LLM -----")
    print(llm.predict("こんにちは"))
    print()

    print("----- Chat Model -----")
    print(chat_model.predict("こんにちは"))
    print()

    print("\033[42mString変数を与えた場合: predict\033[0m")
    text = "カラフルな靴下を作成する会社にふさわしい名前を5つ教えてください"
    print(f"{text=}")

    print("----- LLM -----")
    print(llm.predict(text))
    print()

    print("----- Chat Model -----")
    print(chat_model.predict(text))
    print()

    print("\033[43m配列を与えることのできるメソッド: predict_messages\033[0m")
    text = "カラフルな靴下を作成する会社にふさわしい名前を5つ教えてください"
    print(f"{text=}")
    messages = [HumanMessage(content=text)]
    print(f"Human Message={messages}")

    print("----- LLM -----")
    print(llm.predict_messages(messages))
    print()

    print("----- Chat Model -----")
    print(chat_model.predict_messages(messages))
    print()


def prompt_templates():
    """Prompt Templatesの動きを確認する
    1. Prompt Templatesの出力を確認
    2. `ChatPromptTemplate` に複数の `Prompt Templates` を与えて実際にChatModelに設定する"""

    print("\033[041mPrompt Templates\033[0m")
    template = "What is a good name for a company that makes {product}?"
    print(f"{template=}")
    prompt = PromptTemplate.from_template(template)
    print(f"{prompt.format(product='colorful socks')=}")
    print()

    print("\033[042m複数のPrompt Templatesの活用\033[0m")
    template = (
        "Your are a helpful assistant that translates "
        "{input_language} to {output_language}"
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    complete_prompt = chat_prompt.format_messages(
        input_language="English",
        output_language="Japanese",
        text="I love programming.",
    )
    print(f"{complete_prompt=}")

    chat_model = ChatOpenAI()
    print(chat_model.predict_messages(complete_prompt))


def output_parsers():
    """Output Parsesの確認"""

    class CommaSeparatedListOutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text.strip().split(", ")

    print(CommaSeparatedListOutputParser().parse("hi, bye"))


def llm_chain():
    """LLM Chainの動きを確認する"""

    class CommaSeparatedListOutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text.strip().split(", ")

    template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects
in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=chat_prompt,
        output_parser=CommaSeparatedListOutputParser(),
    )

    print(chain.run("colors"))


if __name__ == "__main__":
    # tutorial_of_llms()
    # prompt_templates()
    # output_parsers()
    llm_chain()

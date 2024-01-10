import langchain
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import Tool

langchain.verbose = True
langchain.debug = True


def get_chat():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def my_super_func(params):
    return 42


if __name__ == "__main__":
    tools = [
        Tool.from_function(
            func=my_super_func,
            name="The_Answer",
            description="生命、宇宙、そして万物について究極の疑問の答え"
        ),
    ]



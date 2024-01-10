from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger


def main():
    """https://python.langchain.com/docs/modules/model_io/models/chat/"""
    load_dotenv(find_dotenv())

    chat = ChatOpenAI(model_name='gpt-3.5-turbo',
                      temperature=0.0,
                      verbose=True)
    logger.info('chat model: {}', chat)

    messages = [
        SystemMessage(content='You are a helpful assistant.'),
        HumanMessage(content='Who won the world series in 2020?'),
    ]
    ai_message = chat(messages)
    logger.info('AI message: {}', ai_message)


if __name__ == '__main__':
    main()

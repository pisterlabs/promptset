import langchain
from dotenv import find_dotenv, load_dotenv
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from loguru import logger


def main():
    load_dotenv(find_dotenv())

    langchain.llm_cache = SQLiteCache('langchain.sqlite')

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0, verbose=True)
    logger.info('chat: {}', chat)

    message = chat.predict('What is the best way to make a pizza?')
    logger.info('message: {}', message)


if __name__ == '__main__':
    main()

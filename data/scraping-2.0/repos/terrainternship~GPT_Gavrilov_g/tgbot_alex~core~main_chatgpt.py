import asyncio
import sys

from logger.logger import logger
from config import ROOT_DIR, SETTINGS_PATH, FAISS_DB_DIR, SYSTEM_PROMT_FILE, USER_PROMT, CHEAP_MODEL, EXPENSIVE_MODEL, TEMPERATURE
from create_bot import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import tiktoken
import re
import os


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
faiss_db_dir = os.path.join(ROOT_DIR, FAISS_DB_DIR)
os.chdir(faiss_db_dir)

# Прочитаем Системный промт из файла
system_promt_file = os.path.join(ROOT_DIR, SETTINGS_PATH, SYSTEM_PROMT_FILE)
try:
    with open(system_promt_file, 'r', encoding='utf-8') as file:
        system_promt = file.read()
        logger.info(f'(Прочитали system-promt)')
        # logger.info(f'(Прочитали system-promt): {system_promt_file} - {system_promt}')
except Exception as e:
    print(f'Ошибка чтения ПРОМТА: {e}')


class WorkerOpenAI:
    def __init__(self, faiss_db_dir=faiss_db_dir, list_indexes=None, mod=EXPENSIVE_MODEL):
        # старт инициализации промта
        # Прочитаем Системный промт из файла
        system_promt_file = os.path.join(ROOT_DIR, SETTINGS_PATH, SYSTEM_PROMT_FILE)
        try:
            with open(system_promt_file, 'r', encoding='utf-8') as file:
                system_promt = file.read()
                logger.info(f'(Прочитали system-promt)')
                # logger.info(f'(Прочитали system-promt): {system_promt_file} - {system_promt}')
        except Exception as e:
            print(f'Ошибка чтения ПРОМТА: {e}')

        # конец инициализации промта

        # Составим список всех индексов в папке faiss_db_dir:
        # print(f'Ищем список курсов: {faiss_db_dir}')
        if list_indexes is None:
            list_indexes = []
            for folder in os.listdir(faiss_db_dir):
                if os.path.isdir(os.path.join(faiss_db_dir, folder)):
                    list_indexes.append(os.path.basename(folder))
        #print(f'__init__: Нашли базы: {list_indexes}')

        self.model = mod
        self.list_indexes = list_indexes
        # системные настройки
        self.chat_manager_system = system_promt

        def create_search_index_old(db_path, indexes):
            flag = True
            # Перебор всех курсов в списке courses:
            # print(f'Старт create_search_index: {indexes =}')
            count_base = 0  # сосчитаем количество курсов
            for index_file in indexes:
                index_path = os.path.join(db_path, index_file)      # получаем полный путь к курсу
                # print(f'create_search_index - ищем индекс {count_base}: {index_file =}, {index_path =}')
                try:
                    # print(f'Пройдемся внутри папки {db_path =}:')
                    for current_base in os.listdir(db_path):    # Перебор всех баз данных в курсе
                        # print(f'Пройдемся внутри папки {db_path =}:')
                        count_base += 1
                        if flag:
                            # Если flag равен True, то загружается база данных FAISS из файла curr_base в папке index_path
                            path_to_current_base = os.path.join(index_path, current_base)
                            db = FAISS.load_local(index_path, OpenAIEmbeddings())
                            flag = False
                        else:
                            # Иначе происходит объединение баз данных FAISS
                            pass
                            #db.merge_from(FAISS.load_local(index_path, OpenAIEmbeddings()))

                    # print(f'Создали индекс {db =}')
                except Exception as e:
                    logger.error(f'1. Ошибка чтения индексов: {e}')
                    sys.exit(777)
            return db

        def create_search_index(indexes):
            """
                Чтение индексов из всех индексных файлов
                :param path: локальный путь в проекте до папки с индексами
                :return: база индексов
                """
            db_path = os.path.join(ROOT_DIR, FAISS_DB_DIR)
            flag = True  # Признак первой базы для чтения. Остальные базы будем добавлять к имеющейся
            # Перебор всех курсов в списке courses:
            # print(f'Старт read_faiss_indexes: {indexes =}')
            count_base = 0  # сосчитаем количество курсов
            for index_file in indexes:
                index_path = os.path.join(db_path, index_file)  # получаем полный путь к курсу
                # print(f'read_faiss_indexes - ищем индекс {count_base}: {index_file =}, {index_path =}')
                count_base += 1
                if flag:
                    # Если flag равен True, то создается база данных FAISS из папки index_path
                    db = FAISS.load_local(index_path, OpenAIEmbeddings())
                    flag = False
                    # print(f'read_faiss_indexes: прочитали новый индекс')
                else:
                    # Иначе происходит объединение баз данных FAISS
                    db.merge_from(FAISS.load_local(index_path, OpenAIEmbeddings()))
                    # print(f'read_faiss_indexes: Добавили в индекс')
            return db

        # Если База данных embedding уже создана ранее
        # print(f'Проверим путь до базы знаний: {faiss_db_dir}')
        if faiss_db_dir:
            # print(f'{os.getcwd() = }')
            # print("Ищем готовую базу данных. Путь: ", faiss_db_dir)
            # print("Курсы: ", self.list_indexes)
            self.search_index = create_search_index(self.list_indexes)

    # пример подсчета токенов
    def num_tokens_from_messages(self, messages):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if self.model == "gpt-3.5-turbo" or "gpt-3.5-turbo-16k":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {self.model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    async def get_chatgpt_answer(self, topic):
        # Выборка документов по схожести с вопросом
        docs = await self.search_index.asimilarity_search(topic, k=8)
        #print(f'get_chatgpt_answer: {docs}')
        message_content = re.sub(r'\n{2}', ' ', '\n '.join(
            [f'\n==  ' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
        user_promt = USER_PROMT.format(message_content, topic)
        messages = [
            {"role": "system", "content": f"{self.chat_manager_system}"},
            {"role": "user", "content": user_promt}
        ]

        # TODO: добавить вторую более дешевую модель. Выбирать модель в зависимости от объема передаваемого user_promt
        completion = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE
        )

        #print(f'{completion["usage"]["total_tokens"]} токенов использовано всего (вопрос-ответ).')
        print('ЦЕНА запроса с ответом :', 0.004*(completion["usage"]["total_tokens"]/1000), ' $')
        #print('===========================================: \n')
        #print('Ответ ChatGPT: ')
        #print(completion.choices[0].message.content)
        return completion, messages, docs


if __name__ == '__main__':
    question = """
    Я прохожу урок Треккинга. Расскажи подробнее про Фильтр Калмана.
    """
    # Создаем объект для дообучения chatGPT
    # Если База данных embedding уже создана ранее
    print(f'{os.path.abspath(faiss_db_dir) = } ')
    curator = WorkerOpenAI(faiss_db_dir=faiss_db_dir)
    answer = asyncio.run(curator.get_chatgpt_answer(question))
    print(answer)

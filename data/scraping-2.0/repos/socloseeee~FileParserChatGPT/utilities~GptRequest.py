"""gpt"""
import time
import sqlite3
from abc import ABCMeta

import g4f
import configparser
from PyQt5.QtCore import pyqtSignal, QThread
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.gigachat import GigaChat
from langchain_core.callbacks import BaseCallbackHandler

config = configparser.ConfigParser()
config.read('credentials.ini')
value1 = config.get('Section1', 'variable1')


class StreamHandler(BaseCallbackHandler):
    def __init__(self, signal):
        self.signal = signal

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)
        self.signal.emit(token, 0)


class GptThread(QThread):
    gpt_result = pyqtSignal(str, int)
    updateDB = pyqtSignal(str, str, int, ABCMeta, dict)

    def __init__(self, text, extension, model, chat_name, isSummarisation, tabIndex, provider, model_dict):
        super().__init__()
        self.model = model
        self.text = text
        self.extension = extension
        self.chat_name = chat_name
        self.tabIndex = tabIndex
        self.provider = provider
        self.model_dict = model_dict
        self.isSummarisation = isSummarisation

    def run(self):
        try:
            if self.text:
                text = self.text
                print(self.model.__name__)

                connection = sqlite3.connect('database/db.sqlite3')  # Replace with your database name
                cursor = connection.cursor()

                # Insert serialized data into the table
                cursor.execute(
                    '''
                    SELECT * 
                    FROM 
                        Chat as c
                        INNER JOIN
                        Messages as m
                        ON c.chat_id = m.chat_id
                    WHERE c.chat_name = ? 
                    LIMIT 10
                    ''', (self.chat_name,)
                )
                print(self.model.__name__)
                if self.model.__name__ == "GigaChat":
                    self.GigachatRun(text, self.isSummarisation, cursor, self.extension, self.tabIndex, self.provider, self.model_dict)
                else:
                    self.OtherModelRun(text, self.model, self.isSummarisation, cursor, self.extension, self.tabIndex, self.provider, self.model_dict)
                # self.gpt_result.emit("\n\n", 0)
                # self.updateDB.emit()
        except Exception as e:
            self.gpt_result.emit(str(e), 1)

    def GigachatRun(self, text, summarisation, cursor, ext, tabIndex, provider, model_dict):
        chat = GigaChat(
            credentials=value1,
            scope="GIGACHAT_API_CORP",
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            streaming=True,
            callbacks=[StreamHandler(self.gpt_result)],
        )

        messages = []
        for user_message, bot_message in zip(cursor.fetchall()[::2], cursor.fetchall()[1::2]):
            messages.append(HumanMessage(content=user_message))
            messages.append(AIMessage(content=bot_message))
        messages.append(
            SystemMessage(
                content=f"Давай я буду кидать тебе содержимое текстовых файлов (csv, txt, rtf, docx, "
                        f"xlsx и так далее), а ты будешь анализировать данное содержимое и кратко "
                        f"суммаризировать.\n"
            ),
        )

        messages = [
            HumanMessage(content=f"Просуммаризируй содержимое {ext}-файла на русском:\n" * summarisation + text)
        ]
        # print(messages)
        self.gpt_result.emit(f"\nБот: ", 0)
        content = ''
        for message in chat(messages).content:
            content += message
        self.updateDB.emit(f"Просуммаризируй содержимое {ext}-файла на русском:\n" * summarisation + text, "User", tabIndex, provider, model_dict)
        self.updateDB.emit(content, "Assistant", tabIndex, provider, model_dict)
        self.gpt_result.emit("\n\n", 0)

    def OtherModelRun(self, text, model, summarisation, cursor, ext, tabIndex, provider, model_dict):

        max_retries = 10  # Максимальное количество попыток

        for attempt in range(max_retries):
            try:

                messages = []
                print(cursor.fetchall())
                for user_message, bot_message in zip(cursor.fetchall()[::2], cursor.fetchall()[1::2]):
                    messages.append({"role": "user", "content": user_message})
                    messages.append({"role": "assistant", "content": bot_message})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Просуммаризируй содержимое {ext}-файла:\n" * summarisation + text
                    }
                )
                # print(messages)
                response = g4f.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # gpt-3.5-turbo
                    messages=messages,
                    provider=model,
                    stream=True
                )
                # Продолжаем обработку response
                self.gpt_result.emit("\nБот: ", 0)
                content = ''
                for message in response:
                    content += message
                    self.gpt_result.emit(message, 0)
                self.gpt_result.emit("\n\n", 0)
                self.updateDB.emit(f"Просуммаризируй содержимое {ext}-файла на русском:\n" * summarisation + text,
                                   "User", tabIndex, provider, model_dict)
                self.updateDB.emit(content, "Assistant", tabIndex, provider, model_dict)
                break
            except Exception as e:
                # print(f"Ошибка: {e}")
                if attempt < max_retries - 1:
                    print(f"Повторная попытка через 5 секунд...")
                    self.gpt_result.emit(f"{attempt + 1}-я попытка: {e}\nПовторная попытка через 5 секунд...", 1)
                    time.sleep(5)  # Подождать 30 секунд перед повторной попыткой
                else:
                    self.gpt_result.emit("\nПревышено максимальное количество попыток. Прекращаем.", 1)
                    # print("Превышено максимальное количество попыток. Прекращаем.")
                    break

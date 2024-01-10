import openai
import tiktoken
from datetime import datetime
import requests

# конфиг и файлы с базами знаний
import chatGPT.auchan_config as CFG
import chatGPT.auchan_data.intents as INTENT  # определение интента запроса
import chatGPT.auchan_data.common as COMMON  # начальная настройка
import chatGPT.auchan_data.delivery as DELIVERY  # "условия доставки"
import chatGPT.auchan_data.delivery_alk as DELIVERY_ALK  # "доставка 18+"
import chatGPT.auchan_data.delivery_faq as DELIVERY_FAQ  # "вопросы доставки"
import chatGPT.auchan_data.delivery_free as DELIVERY_FREE  # "бесплатная доставка"
import chatGPT.auchan_data.info as INFO  # "об ашане"
import chatGPT.auchan_data.refund_exceptions as REFUND_EXC  # "возврат исключения"
import chatGPT.auchan_data.refund_food as REFUND_FOOD  # "возврат продовольственного"
import chatGPT.auchan_data.refund_nonfood as REFUND_NONFOOD  # "возврат непродовольственного"
import chatGPT.auchan_data.shops as SHOPS  # "адреса ашан"
# import chatGPT.auchan_data.city as CITY  # Города в РФ

from config import CHAT_GPT_API_KEY, API_URL_EMOTION, API_TOKEN_EMOTION

# from chatGPT.json_function import json_for_logs


class chatGPT():
    def __init__(self, id_user):
        # print(SYSTEM)
        # self.model = "gpt-3.5-turbo"  # 4096 токенов
        # self.model = "gpt-3.5-turbo-16k"  # 16384 токена
        self.model = "gpt-4"  # 8192 токена
        openai.api_key = CHAT_GPT_API_KEY
        self.max_tokens = 1000
        self.temperature = 0.1
        self.token_limit = 8000
        # Статус завершения диалога
        self.end_dialog = False

        # настройка роли асистента для определения темы сообщения
        self.intent_prompt = [
            {"role": "system", "content": INTENT.SYSTEM},
            {"role": "assistant", "content": INTENT.ASSISTANT}
        ]

        # настраиваем роли и даем базу для ответов
        self.messages = [
            # системная роль, чтобы задать поведение помошника
            {"role": "system", "content": CFG.SYSTEM},
            # промт для выяснения темы проблемы клиента
            {"role": "assistant", "content": COMMON.ASSISTANT}
        ]

        # ID пользователя
        self.id_user = str(id_user)
        # Лог диалога
        self.logs = {
            self.id_user: []
        }
        # Лог переписок
        self.log_dialog = []
        # Диалог и его свойства
        self.dialog = dict

    def emotion_text(self, payload):
        headers = {"Authorization": f"Bearer {API_TOKEN_EMOTION}"}

        self.response_emotion = requests.post(API_URL_EMOTION, headers=headers, json=payload)
        return self.response_emotion.json()

    def review_model(self):
        self.model = openai.Model.list()
        for model in self.model.data:
            print(model.id)

    # функция подсчета числа токенов
    def num_tokens_from_messages(self, messages):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # каждое сообщение следует за <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # если есть имя, то роль опускается
                    num_tokens += -1  # роль всегда обязательна и всегда 1 токен
        num_tokens += 2  # в каждом ответе используется <im_start> помощник
        return num_tokens

    # функция делает запрос и возвращает ответ модели
    def get_response(self, model="gpt-4", msg="", tokens=100, temp=0.1):
        # формируем запрос к модели
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=msg,
            max_tokens=tokens,
            temperature=temp
        )
        # получаем ответ
        chat_response = completion.choices[0].message.content
        return chat_response

    def calculate_dialogue_duration(self, data_json):
        # Подсчет и вывод общего времени для каждого диалога
        for user_id, dialogues in data_json.items():
            for dialogue in dialogues:
                if len(dialogue) < 2:
                    return None

                start_time = datetime.strptime(dialogue[0]['datetime'], "%Y-%m-%d %H:%M:%S.%f")
                end_time = datetime.strptime(dialogue[-1]['datetime'], "%Y-%m-%d %H:%M:%S.%f")
                duration = (end_time - start_time).total_seconds()

                if duration is not None:
                    print(f"ID пользователя: {user_id}, Длительность диалога: {duration} секунд")

                return duration

    def prompt(self, content, voice_file=None, photo_file=None):
        if content == "":
            content = "Привет! Как тебя зовут?"
            # добавляем сообщение пользователя
        print('='*100)
        print(f'''
        ID: {self.id_user}
        Пользователь: {content}
        ''')

        self.messages.append({"role": "user", "content": content})
        self.intent_prompt.append({"role": "user", "content": content})

        # пытаемся получить тему сообщения
        self.intent = self.get_response(model=self.model, msg=self.intent_prompt, tokens=100, temp=0.1)

        # определяем совапдения по темам и загружаем нужный промт
        if "возврат продовольственного" in self.intent:
            self.messages[1] = {"role": "assistant", "content": REFUND_FOOD.ASSISTANT}
        elif "возврат непродовольственного" in self.intent:
            self.messages[1] = {"role": "assistant", "content": REFUND_NONFOOD.ASSISTANT}
        elif "возврат исключения" in self.intent:
            self.messages[1] = {"role": "assistant", "content": REFUND_EXC.ASSISTANT}
        elif "условия доставки" in self.intent:
            self.messages[1] = {"role": "assistant", "content": DELIVERY.ASSISTANT}
        elif "бесплатная доставка" in self.intent:
            self.messages[1] = {"role": "assistant", "content": DELIVERY_FREE.ASSISTANT}
        elif "вопросы доставки" in self.intent:
            self.messages[1] = {"role": "assistant", "content": DELIVERY_FAQ.ASSISTANT}
        elif "доставка 18+" in self.intent:
            self.messages[1] = {"role": "assistant", "content": DELIVERY_ALK.ASSISTANT}
        elif "адреса ашан" in self.intent:
            self.messages[1] = {"role": "assistant", "content": SHOPS.MSW_SCB}
        elif "об ашане" in self.intent:
            self.messages[1] = {"role": "assistant", "content": INFO.ASSISTANT}
        elif "проверка чека" in self.intent:
            self.messages[1] = {"role": "assistant", "content": 'Попросите клиента прислать фото чека для проверки. У Вас есть возможность проверить чек с помощью технологии Yandex Vision. Передаем работу его.'}
        else:
            self.messages[1] = {"role": "assistant", "content": COMMON.ASSISTANT}

        # общее число токенов
        self.conv_history_tokens = self.num_tokens_from_messages(self.messages)
        print('-'*100)
        print(f"Токенов: {self.conv_history_tokens}, интент: {self.intent}")
        print('-'*100)

        # удаляем прошлые сообщения, если число токенов превышает лимиты
        while self.conv_history_tokens + self.max_tokens >= self.token_limit:
            del self.messages[2]
            self.conv_history_tokens = self.num_tokens_from_messages(self.messages)

        # формируем запрос и получаем ответ
        self.chat_response = self.get_response(model=self.model, msg=self.messages, tokens=self.max_tokens,
                                               temp=self.temperature)

        # выводим ответ
        print(f'''
        Ашанчик: {self.chat_response}
        ''')
        print('='*100)

        # сохраняем контекст диалога
        self.messages.append({"role": "assistant", "content": self.chat_response})

        self.dialog = {
            'user': content,
            'chatGPT': self.chat_response,
            'intent': self.intent,
            'common_token': self.conv_history_tokens,
            'datetime': datetime.now(),
            'voice_file': voice_file,
            'photo_file': photo_file
        }

        # Добавляем диалог в логи пользователя
        self.log_dialog.append(self.dialog)

        # Конец диалога
        if "конец диалога" in self.intent:
            self.messages[1] = {"role": "assistant", "content": COMMON.ASSISTANT}
            # Конец измерения времени. Занесение данных в БД
            self.logs[self.id_user].append(self.log_dialog)

            # JS = json_for_logs()

            # JS.merge_data(self.logs, id_user=self.id_user)

            # Лог диалога
            self.logs = {
                self.id_user: []
            }
            # Лог переписок
            self.log_dialog = []
            # Диалог и его свойства
            self.dialog = dict
            # Включаем статус завершения диалога
            self.end_dialog = True


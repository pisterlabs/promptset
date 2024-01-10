import openai
from random import choice
import csv
from datetime import datetime

openai.api_key = 'API_KEY'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]


class HappyBirthday:
    __FLAG=None

    def __new__(cls, *args, **kwargs):
        if HappyBirthday.__FLAG is None:
            HappyBirthday.__FLAG=super().__new__(cls)
            return HappyBirthday.__FLAG
        return HappyBirthday.__FLAG


    def __call__(self, *args, **kwargs):
        if args[0].rsplit('.')[1] == 'csv':
            self.congratulation_file(args[0])
        else:
            if len(args) == 2:
                return ','.join(self.congratulation_name(*args))
            else:
                print(f'Неверный формат данных')

    def congratulation_file(self, *args):
        '''Открытие файла на чтение
         Перебор данных из файла
         открытие файла на запись
          и передача имени и даты в ChatGpt с дальнейшей записью в файл'''
        lst = []
        with open(args[0], newline='') as f:
            file = csv.reader(f, delimiter=' ', quotechar='|')
            for n, i in enumerate(file):
                lst.append(tuple(''.join(i).split(',')))
        with open(args[0], 'w') as user:
            writer = csv.writer(user)
            for n, i in enumerate(lst):
                data, name = i
                if n == 0 and i[0].isalpha() and i[1].isalpha():
                    writer.writerow([data, name])
                else:
                    writer.writerow(self.congratulation_name(data, name))

    def congratulation_name(self, data, name):
        '''Проверка на корректность данных и генерация ответа ChatGpt '''
        if self.check_name(name) and self.check_date(data):
            message = ''.join(choice([
                f"Напиши поздравление с днём рождения для {name}, как аристократ, уложись в 120 символов ответ предоставь на русском"
                f"Напиши поздравление с днём рождения для {name}, сделав акцент на месяц рождения{data[1]} и указав его дату {data[0]}, уложись в 120 символов ответ предоставь на русском",
                f"Напиши поздравление с днём рождения для {name} как чёткий пацан, уложись в 120 символов, добавь смайлы в конец поздравления ответ предоставь на русском",
                f"Напиши поздравление с днём рождения для {name} как писатель Есенин, уложись в 120 символов ответ предоставь на русском",
                f"Напиши поздравление с днём рождения для {name} в стиле хоку, уложись в 120 символов ответ предоставь на русском",
                f"Напиши поздравление с днём рождения для {name} по фене, уложись в 120 символов ответ предоставь на русском",
                f"Напиши поздравление с днём рождения для {name} в форме анекдота, уложись в 120 символов, добавь смайлы в конец поздравления ответ предоставь на русском",
            ]))

            messages.append(
                {"role": "user", "content": message},
            )
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = ' '.join(chat_completion.choices[0].message.content.split('\n'))
            messages.append({"role": "assistant", "content": answer})
            return [data, name, answer]
        return [data, name, f'Неправильный формат даты']

    @staticmethod
    def check_date(x: str):
        '''Проверка даты на корректность ввода'''
        try:
            datetime.strptime(x, "%d.%m")
            return True
        except Exception:
            raise ValueError('Неверный формат даты')

    @staticmethod
    def check_name(x):
        '''Проверка имени на корректность ввода'''
        if all((i.isalpha() for i in x)):
            return True
        raise ValueError('Неверный формат имени')


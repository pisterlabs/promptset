from colorama import Fore, init
import json
import os 
import openai
from openai.error import AuthenticationError, ServiceUnavailableError, APIConnectionError

from utilities import commands_parser
from data_storage import DATA_DIRECTORY

init()

def gpt_response(prompt, story,):
# Define a conversation with the model
    conversation = [
        {
        "role": "system", 
        "content": story
            },                               
        {
            "role": "user", 
            "content": prompt
        },
    ]

    # Send the conversation to the model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Get the model's reply
    reply = response['choices'][0]['message']['content']
    if len(reply) != 0:
        return reply
    else:
        return False
    
def gpt_answer(args: list, story: str) -> None:
    prompt = ' '.join(args)
    print('Зачекайте, будь ласка, готую вам найкращу відповідь...')
    response = gpt_response(prompt, story)
    
    print(f'>>> (stepan): {response}')

def set_story(args: list) -> None:
    story = ' '.join(args)
    return story

def greeting():
    print()
    print(Fore.BLUE + f'{" "*5}Вас вітає ваш персональний GPT ПОМІЧНИК 🦾')
    print(Fore.YELLOW + f'{" "*5}Тут ви можете задавати хвилюючі запитання та отримувати вичерпні відповіді' + Fore.WHITE)
    print("""
        Короткий гайд використання бота помічника:         
    спочатку  тобі потрібно ввести коректний API ключ, який можна отримати на сайті: https://platform.openai.com/account/api-keys, 
    якщо немає коштів на балансі - бот працювати не буде. Якщо ключ введено або вказано невірно - бот запросить ввести 
    його по-новому. Далі потрібно буде ввести питання, що цікавить тебе, і бот з радістю відповість на всі питання) Вихід з бота 
    здійснюється командами exit, good byе, close')
""")

COMMANDS = {
    'story': set_story
}

END_COMMANS = ['exit', 'good bye', 'close']



first_start = True

def gpt_app():
    try:
        file_path = DATA_DIRECTORY / 'key.bin'

        global first_start
        if first_start:
            greeting()
            first_start = False
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                encoded_key = fh.read()
                openai.api_key = encoded_key.decode('utf-16').strip('"')
        else:
            key = input('Будь ласка, введіть ключ AI:')
            if key:
                choose = input('Ви хочете зберегти свій ключ API?, введіть (Y/n):')
                if choose != 'n':
                    with open(file_path, 'wb') as fh:
                        encoded_key = key.encode('utf-16')
                        fh.write(encoded_key)
                    openai.api_key = key
                    print('Ваш ключ успiшно збережено')
                else:
                    openai.api_key = key
            else:
                gpt_app()

        is_working = True
        while is_working:
            story = '''You act like ukrainian national hero Stepan Bandera. Before every answer you say "Glory to Ukraine". Your aswer as short as 
            posible and at the same time informative'''
            user_input = input('>>> (you): ')

            command, arguments = commands_parser(user_input)
            if command in COMMANDS:
                command_handler = COMMANDS[command]
                story = command_handler(arguments)
                print(story)
            elif command in END_COMMANS:
                print('Сподіваюсь, ти скоро повернешся! Не забувай контролювати свій баланс.')
                first_start = True
                is_working = False
            else:
                gpt_answer(arguments, story)

    except KeyboardInterrupt:
        print('\nБудь ласка, користуйся командами для завершення роботи боту.')
        first_start = True
    except (AuthenticationError, UnicodeEncodeError):
        print('Упс... відбулася помилка аутентифікації. Будь ласка, введи коректний API ключ, який можна отримати на сайті: https://platform.openai.com/account/api-keys')
        if os.path.exists(file_path):
            os.remove(file_path)
            gpt_app()
        else:
            gpt_app()
    except ServiceUnavailableError:
        print('На даний момент сервер переповнений. Будь ласка, повтори спробу пізніше або перевір з\'єднання з Інтернетом.')
        gpt_app()
    except APIConnectionError:
        print("Виникла проблема зі з'єднання. Будь ласка перевірте своє підключення до інтернету чи спробуйте пізніше.")
        gpt_app()

if __name__ == "__main__":
    gpt_app()

                    
import os
import io
import sys
import time
import sqlite3
import base64
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam,  ChatCompletionMessageParam, ChatCompletionUserMessageParam
from PIL import Image
from rich import print
from rich.status import Status


from .settings import settings, chat_settings


load_dotenv('.env.local')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    OPENAI_API_KEY = input('Enter your OpenAI API key: ')


ai = OpenAI(api_key=OPENAI_API_KEY)
db = sqlite3.connect('data.db')


class State:
    def __init__(self):
        self.__mode: Literal['chat', 'image'] = 'chat'
        self.__chat_id: int | None = None
        self.__messages: list[ChatCompletionMessageParam] = []

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value: Literal['chat', 'image']):
        self.__mode = value

    @property
    def chat_id(self):
        return self.__chat_id

    def new_chat(self):
        self.__chat_id = insert_chat()

    def reset_chat(self):
        self.__chat_id = None
        self.__messages = []

    @property
    def messages(self):
        return self.__messages

    def add_message(self, message: ChatCompletionMessageParam):
        self.__messages.append(message)
        insert_message(message)


state = State()


def create_tables():
    db.execute('''
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER,
            updated_at INTEGER
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS message (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            created_at INTEGER,
            usage TEXT,
            FOREIGN KEY (chat_id) REFERENCES chat (id)
        )
    ''')


def insert_chat() -> int:
    cur = db.execute(
        'INSERT INTO chat (created_at) VALUES (?)',
        (int(time.time()),)
    )
    db.commit()

    if cur.lastrowid is None:
        raise Exception('Could not insert chat into database')

    return cur.lastrowid


def insert_message(message: ChatCompletionMessageParam) -> int:
    cur = db.execute(
        'INSERT INTO message (chat_id, role, content, created_at, usage) VALUES (?, ?, ?, ?, ?)',
        (state.chat_id, message['role'], message['content'], int(time.time()), message.get('usage'))
    )
    db.commit()

    if cur.lastrowid is None:
        raise Exception('Could not insert message into database')

    return cur.lastrowid


class Command:
    def help(self):
        show_help()

    def quit(self):
        print('Goodbye!')
        sys.exit(0)

    def clear(self):
        state.reset_chat()
        os.system('cls' if os.name == 'nt' else 'clear')

    def chat(self):
        if state.mode == 'chat':
            print('\n[red]- Chat mode is already active[/] (type "clear" to reset and start a new chat)\n')
            return
        state.mode = 'chat'
        print('[bold green]- MODE:[/] [bold blue]chat[/]\n')

    def image(self):
        if state.mode == 'image':
            print('\n[red]- Image mode is already active[/] (type "clear" to reset and start a new chat)\n')
            return
        state.mode = 'image'
        print('[bold green]- MODE:[/] [bold blue]image[/]\n')

    def set(self, setting: str, value: str):
        if state.mode == 'chat':
            setattr(chat_settings, setting, value)
        elif state.mode == 'image':
            print('TODO')

    def settings(self):
        if state.mode == 'chat':
            print('\n[bold green]- Chat Settings:')
            for k, v in chat_settings.__dict__.items():
                print(f'\t- {k}: [bold blue]{v}[/]')
            print()
        elif state.mode == 'image':
            print('\n[bold green]- Image Settings:')
            print('TODO')


cmd = Command()


def execute(action: str, *args: str) -> None:
    try:
        getattr(cmd, action)(*args)
    except (AttributeError, TypeError, ValueError) as e:
        if isinstance(e, AttributeError):
            print(f'Unknown command: {action}')
        elif isinstance(e, TypeError):
            print(f'Invalid arguments: {args}')
        elif isinstance(e, ValueError):
            print(f'Invalid value: {args}', e)
        print('\n')


def handle_command(prompt: str):
    action, *args = prompt[1:].strip().split()
    execute(action, *args)


def new_prompt() -> ChatCompletionUserMessageParam:
    print('[bold green]> [/]', end='')
    prompt = input()

    if not prompt.strip():
        return new_prompt()

    if prompt.startswith(settings.action_prefix):
        handle_command(prompt)
        return new_prompt()

    return {
        'role': 'user',
        'content': prompt
    }


def show_help() -> None:
    print('\n- Available commands:')
    for cmd in dir(Command):
        if not cmd.startswith('__'):
            print(f'\t- [bold blue]{cmd}[/]')
    print()


def streaming_response() -> ChatCompletionAssistantMessageParam:
    print()
    with Status('[bold blue]Thinking...[/]'):
        stream = ai.chat.completions.create(
            stream=True,
            model=chat_settings.model,
            messages=state.messages
        )

    content = ''

    print('🤖', end=' ')
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='', flush=True)
            content += chunk.choices[0].delta.content
    print('\n')

    return {
        'role': 'assistant',
        'content': content,
    }


def generate_image(prompt: ChatCompletionUserMessageParam) -> None:
    content = str(prompt['content'])
    file_name = f'{content[0:len(content) // 2].replace(' ', '_')}_{int(time.time())}.png'

    output = Path('output')

    with Status('[bold blue]Generating image...[/]'):
        response = ai.images.generate(
            model='dall-e-3',
            prompt=content,
            size='1024x1024',
            n=1,
            response_format='b64_json',
        )

    img = response.data[0]
    if img.b64_json is None:
        raise Exception('Could not generate image')

    data = base64.b64decode(img.b64_json)

    output.mkdir(parents=True, exist_ok=True)
    (output / file_name).write_bytes(data)

    print(f'- [bold green]Image saved to[/] [bold blue]{output / file_name}[/]')

    Image.open(io.BytesIO(data)).show()


def chat_repl(prompt: ChatCompletionUserMessageParam) -> None:
    if state.chat_id is None:
        state.new_chat()

    state.add_message(prompt)
    response = streaming_response()
    state.add_message(response)


def repl() -> None:
    while 1:
        try:
            prompt = new_prompt()

            if state.mode == 'chat':
                chat_repl(prompt)
                continue

            if state.mode == 'image':
                generate_image(prompt)
                continue

        except KeyboardInterrupt:
            db.close()
            print('\nGoodbye!')
            sys.exit(0)


def main() -> None:
    create_tables()

    print('\nWelcome to the AI Playground!\n')
    print('Type "help" to see a list of available commands.')
    print('Type "quit" or press "Ctrl+C" to exit.\n')
    print('Or just ask me me anything you want!\n')

    repl()

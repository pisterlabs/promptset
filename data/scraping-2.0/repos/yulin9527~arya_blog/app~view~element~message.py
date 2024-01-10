import asyncio
import os
from typing import List, Tuple
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from nicegui import ui

OPENAI_API_KEY = os.getenv('OPENAI_KEY', )


def openai_msg():
    llm = ConversationChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo-1106', openai_api_key=OPENAI_API_KEY))

    messages: List[Tuple[str, str]] = []
    thinking: bool = False

    @ui.refreshable
    def chat_messages() -> None:
        for name, _text in messages:
            if name == 'arya':
                ui.chat_message(text=_text, name=name, sent=name == 'arya').classes('w-full')
            else:
                with ui.chat_message(name=name, sent=name == 'arya').classes('w-full'):
                    ui.markdown(_text)
        area.scroll_to(percent=1)
        if thinking:
            ui.spinner(size='3rem').classes('w-full self-center')
        area.scroll_to(percent=1)


    async def send() -> None:
        nonlocal thinking
        message = text.value
        if not message:
            return
        messages.append(('arya', text.value))
        thinking = True
        text.value = ''
        chat_messages.refresh()
        response = await llm.arun(message, )
        messages.append(('娜娜', response))
        thinking = False
        chat_messages.refresh()

    with ui.scroll_area().classes(
            'w-full max-w-2xl mx-auto flex-grow items-stretch') as area:
        chat_messages()

    with ui.row().classes('w-full no-wrap items-center'):
        placeholder = 'message' if OPENAI_API_KEY != 'not-set' else \
            'Please provide your OPENAI key in the Python script first!'
        text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
            .classes('w-full self-center').on('keydown.enter', send)

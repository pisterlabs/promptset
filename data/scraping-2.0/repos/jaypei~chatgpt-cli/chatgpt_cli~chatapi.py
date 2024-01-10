# -*- coding: utf-8 -*-
#
# Copyright 2023, JayPei <jaypei97159@gmail.com>
#


import datetime
import enum
import typing as t

import attrs
import openai
from rich.live import Live
from rich.markdown import Markdown

from chatgpt_cli import config, term
from chatgpt_cli.error import CommandError


# define a enum type for ChatMessageType
class ChatMessageType(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@attrs.define
class ChatMessage:
    message : str
    message_type : ChatMessageType = attrs.field(validator=attrs.validators.in_(ChatMessageType))
    timestamp : int = attrs.field(factory=lambda: int(datetime.datetime.now().timestamp()))

    def to_message_json(self) -> dict:
        return {
            "role": self.message_type.value,
            "content": self.message,
        }


class ChatSession:

    histories : t.List[ChatMessage] = attrs.field(factory=list)

    def __init__(self, session_name: str, prompt: str):
        self.session_name : str = session_name
        self.prompt : str = prompt
        self.histories : t.List[ChatMessage] = []
        self.conversation_count : int = 0
        self.no_context : bool = not config.get_config().getboolean('CLI', 'default_enable_context')

    def __str__(self):
        return f"ChatSession({self.session_name}, {self.prompt}, {self.no_context})"

    def generate_query_messages(self) -> t.List:
        query_messages = []
        if self.no_context:
            # reversed query for the first user message
            for message in reversed(self.histories):
                if message.message_type == ChatMessageType.USER:
                    query_messages.append(message.to_message_json())
                    break
            return query_messages
        for message in self.histories:
            query_messages.append(message.to_message_json())
        return query_messages

    def add_message(self, message: ChatMessage):
        if self.prompt and message.message_type == ChatMessageType.USER:
            prompt_message = config.get_prompt_message(self.prompt)
            if prompt_message:
                message.message = f"{prompt_message}\n\n{message.message}"
        self.histories.append(message)
        if message.message_type == ChatMessageType.USER:
            self.conversation_count += 1


class ChatSessionManager:

    def __init__(self):
        self.sessions : t.Dict[str, ChatSession] = {}
        self.current_session : ChatSession

    def get_session(self, session_name: str) -> ChatSession:
        if session_name not in self.sessions:
            self.sessions[session_name] = ChatSession(
                session_name, prompt=config.get_config()['CLI']['default_prompt'])
        return self.sessions[session_name]

    def switch(self, session_name: str) -> ChatSession:
        self.current_session = self.get_session(session_name)
        return self.current_session

    def rename(self, old_name: str, new_name: str):
        if old_name in self.sessions:
            self.sessions[new_name] = self.sessions[old_name]
            del self.sessions[old_name]

    def create(
        self, session_name: str, auto_switch: bool=True,
        prompt: t.Optional[str]=None
    ) -> ChatSession:
        if prompt is None:
            prompt = config.get_config()['CLI']['default_prompt']
        new_session = ChatSession(session_name, prompt)
        self.sessions[session_name] = new_session
        if auto_switch:
            self.switch(session_name)
        return new_session

    def _new_chat_completion(self, stream: bool) -> openai.ChatCompletion:
        try:
            response = openai.ChatCompletion.create(
                model=config.get_config().get('API', 'CHATGPT_MODEL'),
                messages=self.current_session.generate_query_messages(),
                temperature=int(config.get_config().get('API', 'TEMPERATURE')),
                top_p=1,
                n=1,
                presence_penalty=0,
                frequency_penalty=0,
                stream=stream,
            )
            return response
        except openai.error.RateLimitError as e:
            term.console.print(f"[bold red]Rate limit exceeded: {e}[/bold red]")
            raise CommandError("Rate limit exceeded", 2)

    def _single_output(self, console: term.Console, response: openai.ChatCompletion) -> str:
        message = response['choices'][0]['message']["content"]
        console.print(Markdown(message))
        return message

    def _live_output(self, console: term.Console, response: openai.ChatCompletion) -> str:
        output = []
        with Live(console=console) as live:
            for chunk in response:
                delta_obj = chunk['choices'][0]['delta']
                content = delta_obj.get("content")
                if content is None:
                    continue
                output.append(content)
                md = Markdown("".join(output))
                live.update(md, refresh=True)
        return "".join(output)

    def ask(self, question: str, stream: bool, console: term.Console) -> str:
        self.current_session.add_message(ChatMessage(
            message=question,
            message_type=ChatMessageType.USER,
        ))
        response : openai.ChatCompletion
        with term.make_progress_bar(console) as progress:
            progress.add_task(":thinking_face: [green]Thinking ...", total=None)
            response = self._new_chat_completion(stream=stream)
            answer = ""
        if not stream:
            answer = self._single_output(console, response)
        else:
            answer = self._live_output(console, response)
        self.current_session.add_message(ChatMessage(
            message=answer,
            message_type=ChatMessageType.ASSISTANT,
        ))
        return answer


_session_manager : ChatSessionManager


def get_session_manager() -> ChatSessionManager:
    global _session_manager
    return _session_manager


def init():
    global _session_manager
    openai.api_key = config.get_config().get('API', 'OPENAI_API_KEY')
    _session_manager = ChatSessionManager()
    _session_manager.create('Chat01', auto_switch=True)


def is_valid_openai_api_key(key: str) -> bool:
    if not key.startswith('sk-'):
        return False
    return True

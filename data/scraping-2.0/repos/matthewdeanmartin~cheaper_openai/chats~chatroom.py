"""
A chatroom is a markdown document with the following format:

# Chatroom Name
## User:
text

## Bot Name:
text

## Other Bot Name:
(Message to Bot)

Response from bot
"""
from typing import Optional

from openai.types.beta.threads import ThreadMessage

from chats.bot_shell import Bot


class ChatroomLog:
    def __init__(self, name, thread_id):
        self.name = name.replace(" ", "_").lower()
        self.messages = []

        # TODO: there is a thread_id per bot, not per chatroom
        # so right now this is the thread of the primary bot
        self.filename = f"{name}_{thread_id}.md"

    def write_header(self, main_bot: Bot) -> None:
        with open(self.filename, "a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"## {main_bot.assistant.name} ({main_bot.model}):\n" f"{main_bot.assistant.instructions}\n\n")

    def add_starting_user_message(self, message: ThreadMessage):
        self.messages.append(message)
        with open(self.filename, "a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(f"## User:\n{message.content[0].text.value}\n\n")

    def add_bot_message(self, bot: Bot, message: ThreadMessage, message_to_bot: Optional[ThreadMessage] = None):
        self.messages.append(message)
        with open(self.filename, "a", encoding="utf-8", errors="backslashreplace") as f:
            if message_to_bot:
                prompt = f"({message_to_bot.content[0].text.value})\n\n"
            else:
                prompt = ""
            f.write(f"\n## {bot.assistant.name}:\n" f"{prompt}" f"\n{message.content[0].text.value}\n")

    def add_python_exception(self, exception: Exception, message: str):
        self.messages.append(message)
        with open(self.filename, "a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(f"## Exception:\n" f"{exception}\n" f"{message}\n\n")

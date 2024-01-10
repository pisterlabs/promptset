from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from telegram.ext import ContextTypes

from clients.open_ai.client import OpenaiClient


@dataclass
class BaseBotService:
    open_ai_client: OpenaiClient
    context: ContextTypes.DEFAULT_TYPE

    async def send_photo(self, chat_id: str | int, photo_name: str, description: str) -> None:
        await self.context.bot.send_photo(chat_id=chat_id, photo=photo_name)  # type: ignore
        await self.send_message(chat_id=chat_id, text=description)  # type: ignore

    async def send_message(self, chat_id: int | str, text: str):
        await self.context.bot.send_message(chat_id=chat_id, text=text)

    @classmethod
    def get_photo_name(cls) -> str:
        return f"{uuid4()}.jpg"

    @classmethod
    def get_autoresponse(cls, user_name: Optional[str]) -> str:
        return user_name if user_name else "дружище"

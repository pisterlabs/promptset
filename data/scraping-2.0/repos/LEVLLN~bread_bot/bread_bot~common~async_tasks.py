import random

import procrastinate

from bread_bot.common.clients.openai_client import DallEPrompt
from bread_bot.common.schemas.bread_bot_answers import TextAnswerSchema, PhotoAnswerSchema
from bread_bot.common.services.messages.message_sender import MessageSender
from bread_bot.common.services.think_service import OpenAIService
from bread_bot.common.utils.structs import BOT_NAME
from bread_bot.main.procrastinate import app


@app.task(
    retry=procrastinate.RetryStrategy(
        max_attempts=5,
        wait=3,
    )
)
async def async_free_promt(text: str, chat_id: int, reply_to_message_id: int):
    await MessageSender(
        message=TextAnswerSchema(
            text=await OpenAIService().free_prompt(text),
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )
    ).send_messages_to_chat()


@app.task(
    retry=procrastinate.RetryStrategy(
        max_attempts=5,
        wait=3,
    )
)
async def async_think_about(pre_promt: str, text: str, chat_id: int, reply_to_message_id: int):
    await MessageSender(
        message=TextAnswerSchema(
            text=await OpenAIService().think_about(pre_promt, text),
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )
    ).send_messages_to_chat()


@app.task(
    retry=procrastinate.RetryStrategy(
        max_attempts=5,
        wait=3,
    )
)
async def async_imagine(
    chat_id: int,
    reply_to_message_id: int,
    prompt: str,
    count_variants: int,
):
    images = await OpenAIService().imagine(DallEPrompt(prompt=prompt, count_variants=count_variants))
    await MessageSender(
        message=PhotoAnswerSchema(
            photo=random.choice(images).url,
            caption=f"{prompt} (c) {BOT_NAME}",
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )
    ).send_messages_to_chat()

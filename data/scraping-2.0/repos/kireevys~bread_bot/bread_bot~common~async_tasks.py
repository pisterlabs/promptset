import procrastinate

from bread_bot.common.schemas.bread_bot_answers import TextAnswerSchema
from bread_bot.common.services.messages.message_sender import MessageSender
from bread_bot.common.services.think_service import OpenAIService
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
            text=await OpenAIService().free_promt(text),
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

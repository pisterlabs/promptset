from telegram import Update
from telegram.ext import ContextTypes
from langchain.chat_models import ChatOpenAI
from bullshit_bot_bot_bot.middleware import GenericMessage, with_messages_processed
from bullshit_bot_bot_bot.utils import print_messages

# /conflicts. Have you considered who has an interest in you believing this information and why?
# Make a call to AI asking: Tell me about any types of companies, stakeholders or financial interests that might be interested in sponsoring this piece of text. Limit your answer to 50 words and provide it as a list
# Output in text.


def get_prompt(messages: list[GenericMessage]):
    message_prompt_section = print_messages(messages)

    return f"""
Tell me about any types of companies, stakeholders or financial interests that might be interested in sponsoring this piece of text. Limit your answer to 50 words and provide it as a list

Conversation:
{message_prompt_section}
---
""".strip()


@with_messages_processed
async def get_conflicts(
    messages: list[GenericMessage], update: Update, context: ContextTypes.DEFAULT_TYPE
):
    print_messages(messages)
    model = ChatOpenAI()
    prompt = get_prompt(messages)
    result = model.call_as_llm(prompt)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=result)

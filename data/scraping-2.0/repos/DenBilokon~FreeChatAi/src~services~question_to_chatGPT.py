import logging

import openai

from src.conf.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def question_to_ai(question, pdf_text: str):
    openai.api_key = settings.openai_key

    worker = (
        f"Document Text: {pdf_text}\n\n"
        "Your main task is to give answers with Document Text. "
        "Your name is BOB. "
        "Always strive to give concise answers. "
        "Answer in Ukrainian, unless otherwise indicated. "
    )

    messages = [{"role": "system", "content": worker}]

    if len(question) > 4000:  # Перевіряємо обмеження на довжину запиту

        return "Ваше питання занадто довге. Спробуйте скоротити його."
    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=500,
        messages=messages
    )

    total_tokens = response.get("usage").get("total_tokens")
    logger.info(f"Start request to GPT with messages: {messages}")
    logger.info(f'Total tokens: {total_tokens}')

    response_html = f'I: {question}\nFreeDoc-AI: {response.choices[0].message.content.strip()}'

    return response_html

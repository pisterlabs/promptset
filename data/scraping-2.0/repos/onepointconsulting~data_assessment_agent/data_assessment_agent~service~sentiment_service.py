from typing import List

from openai.types.chat.chat_completion import ChatCompletion

from data_assessment_agent.config.log_factory import logger
from data_assessment_agent.config.toml_support import prompts
from data_assessment_agent.model.sentiment import Sentiment
from data_assessment_agent.service.openai_support import (
    create_completion,
    extract_function_call_arguments,
)
from data_assessment_agent.model.sentiment import answer_sentiment_spec


def create_user_message(question: str, answer: str) -> str:
    user_prompt = prompts["sentiment"]["user_message"]
    return user_prompt.format(question=question, answer=answer)


async def get_answer_sentiment(question: str, answer: str) -> str:
    logger.info("Getting sentiment for %s", answer)
    user_message = create_user_message(question, answer)
    system_message = prompts["sentiment"]["system_message"]
    completion = await create_completion(
        system_message, user_message, answer_sentiment_spec
    )
    return await extract_answer_sentiment(completion)


async def extract_answer_sentiment(chat_completion: ChatCompletion) -> str:
    logger.info("Extracting answer sentiment")
    arguments = extract_function_call_arguments(chat_completion)
    key = "sentiment"
    value = arguments.get(key, Sentiment.UNKNOWN)
    logger.info("Sentiment value type %s", type(value))
    return value if isinstance(value, str) else Sentiment.UNKNOWN


if __name__ == "__main__":
    import asyncio
    from data_assessment_agent.test.provider.sentiment_provider import (
        create_sentiment_qa,
        create_sentiment_negative_qa,
    )

    question, answer = create_sentiment_qa()
    print("Sentiment 1: ", asyncio.run(get_answer_sentiment(question, answer)))

    question, answer = create_sentiment_negative_qa()
    print("Sentiment 2: ", asyncio.run(get_answer_sentiment(question, answer)))

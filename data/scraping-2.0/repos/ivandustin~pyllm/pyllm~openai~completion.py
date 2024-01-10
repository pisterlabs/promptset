from random import choice
import logging
import openai

logger = logging.getLogger(__name__)


def completion(prompt, **kwargs):
    response = openai.Completion.create(
        prompt=prompt,
        **kwargs,
    )
    logger.info(f"Parameters\n{kwargs}")
    logger.info(f"Response\n{response}")
    return choice(response["choices"])["text"]

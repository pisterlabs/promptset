import logging
from typing import Callable, Optional

from langchain.chains import ConversationChain


def retry_until_valid(
    retries: int,
    chain: ConversationChain,
    prompt: str,
    validator: Callable[[str], Optional[str]] = lambda x: None,
):
    response = chain.predict(input=prompt)
    reprompt = validator(response)
    if not reprompt:
        return response
    else:
        logging.warning(f"Could not get a response, will reprompt {reprompt}")

    for retry_number in range(retries):
        response = chain.predict(input=reprompt)
        reprompt = validator(response)

        if not reprompt:
            return response
        else:
            logging.warning(f"Could not get a response, will reprompt {reprompt}")

    raise Exception("Couldn't get valid response")

from typing import List, Union
from openai.types.chat.chat_completion import ChatCompletion

from data_assessment_agent.config.toml_support import prompts
from data_assessment_agent.model.assessment_framework import (
    SuggestedResponseList,
    SuggestedResponse,
    suggested_response_list_spec,
)
from data_assessment_agent.config.log_factory import logger
from data_assessment_agent.service.openai_support import (
    create_completion,
    extract_function_call_arguments,
)

PROMPT_KEY = "suggestion"


def create_user_message(question: str, topic: str) -> str:
    user_prompt = prompts[PROMPT_KEY]["user_message"]
    return user_prompt.format(question=question, topic=topic)


async def generate_suggestions(
    question: str, topic: str
) -> Union[SuggestedResponseList, None]:
    logger.info("Getting suggestions for %s", question)
    user_message = create_user_message(question, topic)
    system_message = prompts[PROMPT_KEY]["system_message"]
    completion = await create_completion(
        system_message, user_message, suggested_response_list_spec
    )
    return await extract_suggested_responses(completion)


async def extract_suggested_responses(
    chat_completion: ChatCompletion,
) -> Union[SuggestedResponseList, None]:
    logger.info("Extracting suggested responses")
    arguments = extract_function_call_arguments(chat_completion)
    key = "suggested_responses"
    try:
        list = arguments.get(key)
        suggested_responses = [
            SuggestedResponse.model_validate(suggested_response_dict)
            for suggested_response_dict in list
        ]
        return SuggestedResponseList(suggested_responses=suggested_responses)
    except:
        logger.exception("Cannot extract suggested response list")


if __name__ == "__main__":
    import asyncio

    question = "What best describes the reality of your organization's advanced analytics tools landscape?"

    res = asyncio.run(generate_suggestions(question))
    print("Suggestions 1: ", res)
    print(type(res))

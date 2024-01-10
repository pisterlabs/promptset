import json
from typing import Optional, List


from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import FunctionCall

from data_assessment_agent.config.log_factory import logger
from data_assessment_agent.config.toml_support import prompts
from data_assessment_agent.config.config import cfg
from data_assessment_agent.model.ranking import (
    question_ranking_spec,
    topic_ranking_spec,
)
from data_assessment_agent.service.openai_support import (
    create_completion,
    extract_function_call_arguments,
)


async def rank_questions(
    topic: str, question_answers: str, ranking_questions: str
) -> List[str]:
    logger.info("Ranking questions")
    user_message = create_user_message(topic, question_answers, ranking_questions)
    system_message = prompts["ranking"]["system_message"]
    completion = await create_completion(
        system_message, user_message, question_ranking_spec
    )
    return extract_ranking(completion)


def extract_ranking(
    chat_completion: Optional[ChatCompletion], ranking_key="ranked_questions"
) -> List[str]:
    arguments = extract_function_call_arguments(chat_completion)
    if ranking_key in arguments:
        return arguments[ranking_key]
    return []


def create_user_message(
    topic: str, question_answers: str, ranking_questions: str
) -> str:
    user_prompt = prompts["ranking"]["user_message"]
    return user_prompt.format(
        topic=topic,
        question_answers=question_answers,
        ranking_questions=ranking_questions,
    )


async def rank_topics(question_answers: str, ranking_topics_str: str) -> List[str]:
    logger.info("Ranking topics")
    user_message = create_topics_user_message(question_answers, ranking_topics_str)
    system_message = prompts["ranking"]["topics"]["system_message"]
    completion = await create_completion(
        system_message, user_message, topic_ranking_spec
    )
    return extract_ranking(completion, ranking_key="ranked_topics")


def create_topics_user_message(question_answers: str, ranking_topics_str: str) -> str:
    user_prompt = prompts["ranking"]["topics"]["user_message"]
    return user_prompt.format(
        question_answers=question_answers, ranking_topics=ranking_topics_str
    )


if __name__ == "__main__":
    import asyncio
    from data_assessment_agent.test.provider.ranking_prompt_provider import (
        ranking_prompt_provider,
        topics_ranking_prompt_provider,
    )

    def test_question_ranking():
        topic, question_answers, ranking_questions = ranking_prompt_provider()
        ranked_questions = asyncio.run(
            rank_questions(topic, question_answers, ranking_questions)
        )
        for i, question in enumerate(ranked_questions):
            print(i, question)

    def test_topic_ranking():
        question_answers, ranking_topics_str = topics_ranking_prompt_provider()
        ranked_topics = asyncio.run(rank_topics(question_answers, ranking_topics_str))
        for i, topic in enumerate(ranked_topics):
            print(i, topic)

    test_question_ranking()
    test_topic_ranking()

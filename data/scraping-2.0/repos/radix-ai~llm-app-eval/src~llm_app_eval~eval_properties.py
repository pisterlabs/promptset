from functools import lru_cache
from typing import Optional

import numpy as np
import openai
from pydantic import BaseModel

from llm_app_eval.evaluator import EvalProperty, PropertyResult, TestCase
from llm_app_eval.llm_app import OutputFormat

PROPERTY_LLM = "gpt-3.5-turbo-0613"


@lru_cache
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def output_similarity(
    test_case: TestCase, llm_app_result: OutputFormat
) -> Optional[PropertyResult]:
    if test_case.reference_output and llm_app_result.answer:
        app_output_emb = get_embedding(llm_app_result.answer)
        reference_emb = get_embedding(test_case.reference_output.answer)
        result = PropertyResult(
            feedback="",
            score=cosine_similarity(app_output_emb, reference_emb),
        )
    else:
        result = None
    return result


def output_verbosity(test_case: TestCase, llm_app_result: OutputFormat) -> Optional[PropertyResult]:
    if test_case.reference_output and llm_app_result.answer:
        result = PropertyResult(
            feedback="", score=len(llm_app_result.answer) / len(test_case.reference_output.answer)
        )
    else:
        result = None
    return result


class LlmPropertyResult(BaseModel):
    feedback: str
    pass_fail: bool


def evaluate_property_with_llm(
    model: str, system_message: str, user_message: str
) -> PropertyResult:
    return openai.ChatCompletion.create(
        model=model,
        response_model=LlmPropertyResult,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )


def factually_consistent(
    test_case: TestCase, llm_app_result: OutputFormat
) -> Optional[PropertyResult]:
    if test_case.reference_output and llm_app_result.answer:
        result = evaluate_property_with_llm(
            model=PROPERTY_LLM,
            system_message="Evaluate the answer. The answer should be factually consistent with the reference answer. If not, explain why.",
            user_message=f"Answer: {llm_app_result.answer}\nReference Answer: {test_case.reference_output.answer}",
        )
    else:
        result = None
    return result


def improves_historical_answer(
    test_case: TestCase, llm_app_result: OutputFormat
) -> Optional[PropertyResult]:
    if test_case.test_input and test_case.historical_output and llm_app_result.answer:
        result = evaluate_property_with_llm(
            model=PROPERTY_LLM,
            system_message="Evaluate the new answer. Is the new answer better than the old answer? Explain why.",
            user_message=f"Question: {test_case.test_input.question}\nOld answer: {test_case.historical_output.answer}\nNew answer: {llm_app_result.answer}",
        )
    else:
        result = None
    return result


def takes_feedback_into_account(
    test_case: TestCase, llm_app_result: OutputFormat
) -> Optional[PropertyResult]:
    if (
        test_case.test_input
        and test_case.historical_output
        and llm_app_result.answer
        and test_case.historical_feedback
    ):
        result = evaluate_property_with_llm(
            model=PROPERTY_LLM,
            system_message="Evaluate the new answer. Does the new answer improve upon the old one by taking the feedback into account? Explain why.",
            user_message=f"Question: {test_case.test_input.question}\nOld answer: {test_case.historical_output.answer}\nOld feedback: {test_case.historical_feedback}\nNew answer: {llm_app_result.answer}",
        )
    else:
        result = None
    return result


def length_within_bounds(
    test_case: TestCase, llm_app_result: OutputFormat
) -> Optional[PropertyResult]:
    if test_case.reference_output and llm_app_result.answer:
        if len(llm_app_result.answer) <= 1.2 * len(test_case.reference_output.answer):
            result = PropertyResult(feedback="The answer is not too long.", score=1)
        else:
            result = PropertyResult(feedback="The answer is too long.", score=0)
    else:
        result = None
    return result


properties = [
    EvalProperty(
        property_name="FactuallyConsistent",
        description="The answer is factually consistent with the reference answer.",
        eval_func=factually_consistent,
    ),
    # EvalProperty(
    #     property_name="CorrectLanguage"
    #     description="The answer is in the same language as the question.",
    # ),
    EvalProperty(
        property_name="ImprovesHistoricalAnswer",
        description="The answer improves upon the historical answer. It is more complete, more concise, or more accurate.",
        eval_func=improves_historical_answer,
    ),
    EvalProperty(
        property_name="TakesFeedbackIntoAccount",
        description="The answer improves upon the historical answer by taking the feedback into account.",
        eval_func=takes_feedback_into_account,
    ),
    EvalProperty(
        property_name="LengthWithinBounds",
        description="The answer is max 20% longer than the reference answer.",
        eval_func=length_within_bounds,
    ),
    EvalProperty(
        property_name="CosineSimilarity",
        description="The answer is similar to the reference answer.",
        eval_func=output_similarity,
    ),
    EvalProperty(
        property_name="Verbosity",
        description="The answer is not too verbose.",
        eval_func=output_verbosity,
    ),
]

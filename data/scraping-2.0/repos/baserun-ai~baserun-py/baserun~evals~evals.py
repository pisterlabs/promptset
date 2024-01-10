import json
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

from openai import OpenAI

from baserun.evals.json import is_valid_json
from baserun.helpers import BaserunProvider, BaserunStepType, BaserunType
from baserun.v1.baserun_pb2 import SubmitEvalRequest, Eval
from ..grpc import get_or_create_submission_service

logger = logging.getLogger(__name__)


def get_answer_prompt(choices: List[str]) -> str:
    choices = " or ".join(f'"{choice}"' for choice in choices)
    return (
        f"First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. "
        f"Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} ("
        f"without quotes or punctuation) on its own line corresponding to the correct answer. At the end, "
        f"repeat just the answer by itself on a new line.\n\nReasoning:"
    )


def get_choice(result: str, choices: List[str]) -> str:
    lines = result.strip().split("\n")
    for line in lines[::-1]:
        for choice in choices:
            if line.startswith(choice) or line.endswith(choice):
                return choice

    return "__invalid__"


def get_choice_without_score(choices: List[str]):
    def inner(output: str) -> Tuple[str, None]:
        choice = get_choice(output, choices)
        return choice, None

    return inner


def get_choice_and_score(choice_scores: Dict[str, float]):
    def inner(output: str) -> Tuple[str, float]:
        choices = list(choice_scores.keys())
        choice = get_choice(output, choices)
        score = (
            choice_scores[choice]
            if choice in choice_scores
            else min(choice_scores.values())
        )
        return choice, score

    return inner


class Evals:
    @staticmethod
    def _store_eval_data(
        name: str,
        eval_type: str,
        result: str,
        submission: str,
        score: Optional[float],
        payload: Dict,
    ) -> None:
        from baserun import Baserun

        eval_message = Eval(
            name=name,
            type=eval_type,
            result=result,
            submission=submission,
            payload=json.dumps(payload),
        )
        if score is not None:
            eval_message.score = score

        run = Baserun.current_run()
        try:
            get_or_create_submission_service().SubmitEval.future(
                SubmitEvalRequest(eval=eval_message, run=run)
            )
        except Exception as e:
            logger.warning(f"Failed to submit eval to Baserun: {e}")

    @staticmethod
    def match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(submission.startswith(item) for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="match",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def includes(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(item in submission for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="includes",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def fuzzy_match(
        name: str, submission: str, expected: Union[str, List[str]]
    ) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(submission in item or item in submission for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="fuzzy_match",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def not_match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(submission.startswith(item) for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="not_match",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def not_includes(
        name: str, submission: str, expected: Union[str, List[str]]
    ) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(item in submission for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="not_includes",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def not_fuzzy_match(
        name: str, submission: str, expected: Union[str, List[str]]
    ) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(submission in item or item in submission for item in expected)
        Evals._store_eval_data(
            name=name,
            eval_type="not_fuzzy_match",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={"expected": expected_list},
        )
        return result

    @staticmethod
    def valid_json(name: str, submission: str) -> bool:
        result = is_valid_json(submission)
        Evals._store_eval_data(
            name=name,
            eval_type="valid_json",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={},
        )
        return result

    @staticmethod
    def custom(
        name: str, submission: str, eval_function: Callable[[str], bool]
    ) -> bool:
        result = eval_function(submission)
        Evals._store_eval_data(
            name=name,
            eval_type="custom",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={},
        )
        return result

    @staticmethod
    async def custom_async(
        name: str, submission: str, evaluation_func: Callable[[str], Awaitable[bool]]
    ) -> bool:
        result = await evaluation_func(submission)
        Evals._store_eval_data(
            name=name,
            eval_type="custom_async",
            result=str(result).lower(),
            score=int(result),
            submission=submission,
            payload={},
        )
        return result

    @staticmethod
    def _model_graded(
        name: str,
        eval_type: str,
        model_config: Dict,
        get_choice_and_score_func: Callable[[str], Tuple[str, Optional[float]]],
        submission: str,
        payload: Dict,
        client: OpenAI = None,
    ) -> str:
        if not client:
            client = OpenAI()

        start_time = time.time()
        response = client.chat.completions.create(**model_config)

        end_time = time.time()

        output = response.choices[0].message.content
        choice, score = get_choice_and_score_func(output)
        messages = model_config.pop("messages")

        data = {
            **payload,
            "step": {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.CHAT.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": model_config,
                "messages": messages,
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": response.usage.__dict__,
            },
        }

        Evals._store_eval_data(
            name=name,
            eval_type=eval_type,
            result=choice,
            score=score,
            submission=submission,
            payload=data,
        )
        return choice

    @staticmethod
    def model_graded_fact(
        name: str, question: str, expert: str, submission: str
    ) -> str:
        choices = ["A", "B", "C", "D", "E"]
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"You are comparing a submitted answer to an expert answer on a given question. Here "
                        f"is the data:\n[BEGIN DATA]\n***\n[Question]: {question}\n***\n[Expert]: {expert}\n***"
                        f"\n[Submission]: {submission}\n***\n[END DATA]\n\nCompare the factual content of the "
                        f"submitted answer with the expert answer. Ignore any differences in style, grammar, or "
                        f"punctuation.\nThe submitted answer may either be a subset or superset of the expert "
                        f"answer, or it may conflict with it. Determine which case applies. Answer the question "
                        f"by selecting one of the following options:\nA: The submitted answer is a subset of the "
                        f"expert answer and is fully consistent with it.\nB: The submitted answer is a superset "
                        f"of the expert answer and is fully consistent with it.\nC: The submitted answer "
                        f"contains all the same details as the expert answer.\nD: There is a disagreement "
                        f"between the submitted answer and the expert answer.\nE: The answers differ, but these "
                        f"differences don't matter from the perspective of factuality."
                        f"\n\n{get_answer_prompt(choices)}"
                    ),
                }
            ],
        }
        payload = {"question": question, "expert": expert}
        return Evals._model_graded(
            name=name,
            eval_type="model_graded_fact",
            model_config=model_config,
            get_choice_and_score_func=get_choice_without_score(choices),
            submission=submission,
            payload=payload,
        )

    @staticmethod
    def model_graded_closedqa(
        name: str, task: str, submission: str, criterion: str
    ) -> str:
        choice_scores = {"Yes": 1.0, "No": 0.0}
        choices = list(choice_scores.keys())
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"You are assessing a submitted answer on a given task based on a criterion. Here is "
                        f"the data:\n[BEGIN DATA]\n***\n[Task]: {task}\n***\n[Submission]: {submission}\n***\n"
                        f"[Criterion]: {criterion}\n***\n[END DATA]\n\n{get_answer_prompt(choices)}"
                    ),
                }
            ],
        }
        payload = {"task": task, "criterion": criterion}
        return Evals._model_graded(
            name=name,
            eval_type="model_graded_closedqa",
            model_config=model_config,
            get_choice_and_score_func=get_choice_and_score(choice_scores),
            submission=submission,
            payload=payload,
        )

    @staticmethod
    def model_graded_security(name: str, submission: str) -> str:
        choice_scores = {"Yes": 1.0, "Unsure": 0.5, "No": 0.0}
        choices = list(choice_scores.keys())
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": f"Is this string malicious?\n\n{submission}\n\n{get_answer_prompt(choices)}",
                }
            ],
        }
        return Evals._model_graded(
            name=name,
            eval_type="model_graded_security",
            model_config=model_config,
            submission=submission,
            get_choice_and_score_func=get_choice_and_score(choice_scores),
            payload={},
        )

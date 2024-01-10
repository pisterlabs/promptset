import os
import time
from typing import Any, Dict

import openai

from higgins.datasets import data_question_datasets
from higgins.nlp import nlp_utils

from . import caching


openai.api_key = os.getenv("OPENAI_API_KEY")


def build_data_question_completion_prompt(
    user_question: str,
    user_data: Dict[str, Any],
    examples: Dict[str, Any],
    task_description: str = None,
) -> str:
    prompt = ""
    if task_description is not None:
        prompt += f"{task_description}"

    for example in examples:
        prompt += f"\n\nDATA\n{example['data']}\n"
        prompt += "\nQUESTIONS"

        for question, answer in example["questions"]:
            prompt += f"\nQ: {question}"
            prompt += f"\nA: {answer} <<END>>"

    prompt += "\n\nDATA\n{data}\n".format(data=user_data)
    prompt += "\nQUESTIONS"
    prompt += "\nQ: {question}".format(question=user_question)
    prompt += "\nA:"
    return prompt


def data_question_completion(
    question: str,
    data: Dict[str, Any],
    engine="davinci",
    cache: Any = None,
):
    prompt = build_data_question_completion_prompt(
        user_question=question,
        user_data=data,
        examples=data_question_datasets.DATA_QUESTION_DATASET_TRAIN,
        task_description="Please answer questions about the data structures below",
    )
    print(prompt)
    # print(f"Prompt: {prompt}")
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        answer = response["choices"][0]["text"].strip()
        cache.add(
            key=cache_key,
            value={
                "question": question,
                "data": data,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def test_data_question_completion():
    for example in data_question_datasets.DATA_QUESTION_DATASET_TEST:
        for question, expected_answer in example["questions"]:
            answer = data_question_completion(
                question=question, data=example["data"]
            )
            if answer != expected_answer:
                print("Answers below do not match ----")
                print(f"Q: {question}\nA: {answer}\nE: {expected_answer}")


if __name__ == "__main__":
    test_data_question_completion()

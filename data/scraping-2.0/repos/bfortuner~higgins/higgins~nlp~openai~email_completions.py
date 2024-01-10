import os
import time
from typing import Any, Dict

import openai

from higgins.datasets import email_datasets
from higgins.nlp import nlp_utils

from . import caching
from . import completion_utils


openai.api_key = os.getenv("OPENAI_API_KEY")


def send_email_completion(cmd: str, engine="davinci-instruct-beta", cache: Any = None):
    prompt = completion_utils.build_completion_prompt(
        question=cmd,
        action_chains=email_datasets.SEND_EMAIL_DATASET_TRAIN,
        task_description="Convert the following text into commands",
    )
    # print(f"Prompt: {prompt}")
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def compose_email_completion(
    cmd: str,
    engine="davinci",
    cache: Any = None,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    prompt = completion_utils.build_completion_prompt(
        question=cmd,
        action_chains=email_datasets.COMPOSE_EMAIL_DATASET_TRAIN,
        task_description="Compose emails from the following notes",
    )
    start = time.time()
    response = openai.Completion.create(
        engine=engine,
        model=None,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<<END>>"],
    )
    # print(f"Time: {time.time() - start:.2f}")
    answer = response["choices"][0]["text"].strip()
    return answer


def build_edit_email_completion_prompt(
    user_text: str,
    first_draft: str,
    feedback: str,
    examples: Dict[str, Any],
    task_description: str = None,
) -> str:
    prompt = ""
    if task_description is not None:
        prompt += f"{task_description}"

    for example in examples:
        params = example["actions"][0]["params"]
        prompt += f"\n\nINPUT\n{params['user_text']}"
        prompt += f"\n\nFIRST DRAFT\n{params['first_draft']}"
        prompt += f"\n\nFEEDBACK\n{params['feedback']}"
        prompt += f"\n\nREVISED\n{example['revised_email']} <<END>>"

    prompt += f"\n\nINPUT\n{user_text}"
    prompt += f"\n\nFIRST DRAFT\n{first_draft}"
    prompt += f"\n\nFEEDBACK\n{feedback}"
    prompt += "\n\nREVISED"
    return prompt


def edit_email_completion(
    user_text: str,
    first_draft: str,
    feedback: str,
    engine="davinci",
    cache: Any = None,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    prompt = build_edit_email_completion_prompt(
        user_text,
        first_draft,
        feedback,
        examples=email_datasets.EDIT_EMAIL_DATASET_TRAIN,
        task_description="Revise the email first draft based on user feedback",
    )
    # print(f"Prompt: {prompt}")
    start = time.time()
    response = openai.Completion.create(
        engine=engine,
        model=None,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<<END>>"],
    )
    # print(f"Time: {time.time() - start:.2f}")
    answer = response["choices"][0]["text"].strip()
    return answer


def build_summarize_email_completion_prompt(
    email_body: str,
    examples: Dict[str, Any],
    task_description: str = None,
) -> str:
    prompt = ""
    if task_description is not None:
        prompt += f"{task_description}"

    for example in examples:
        params = example["actions"][0]["params"]
        prompt += f"\n\nEMAIL\n{params['plain']}"
        prompt += f"\n\nSUMMARY\n{example['summary']} <<END>>"

    prompt += f"\n\nEMAIL\n{email_body}"
    prompt += "\n\nSUMMARY"
    return prompt


def summarize_email_completion(
    email_body: str,
    engine="davinci",
    cache: Any = None,
    temperature: float = 0.3,
    max_tokens: int = 30,
):
    prompt = build_summarize_email_completion_prompt(
        email_body,
        examples=email_datasets.COMPOSE_EMAIL_DATASET_TRAIN,
        task_description="Summarize the following emails",
    )
    # print(f"Prompt: {prompt}")
    start = time.time()
    response = openai.Completion.create(
        engine=engine,
        model=None,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=.1,
        presence_penalty=.1,
        stop=["<<END>>"],
    )
    # print(f"Time: {time.time() - start:.2f}")
    answer = response["choices"][0]["text"].strip()
    return answer


def search_email_completion(cmd: str, engine="davinci-instruct-beta", cache: Any = None):
    prompt = completion_utils.build_completion_prompt(
        question=cmd,
        action_chains=email_datasets.SEARCH_EMAIL_DATASET_TRAIN,
        task_description="Convert the following text into commands",
    )
    # print(f"Prompt: {prompt}")
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def test_send_email_completion():
    for example in email_datasets.SEND_EMAIL_DATASET_TEST:
        answer = send_email_completion(example["query"])
        actions = completion_utils.convert_string_to_action_chain(answer)
        if actions != example['actions']:
            # print(f"Actions does not match {example['actions']}")
            print(f"Q: {example['query']}\nA: {answer}\nI: {actions}\nE: {example['actions']}")


def test_compose_email_completion():
    for example in email_datasets.COMPOSE_EMAIL_DATASET_TEST:
        answer = compose_email_completion(example["query"])
        actions = completion_utils.convert_string_to_action_chain(answer)
        if actions != example['actions']:
            # print(f"Actions do not match {example['actions']}")
            print(f"Q: {example['query']}\nA: {answer}\nI: {actions}\nE: {example['actions']}")


def test_search_email_completion():
    for example in email_datasets.SEARCH_EMAIL_DATASET_TEST:
        answer = search_email_completion(example["query"])
        actions = completion_utils.convert_string_to_action_chain(answer)
        if actions != example['actions']:
            # print(f"Actions does not match {example['actions']}")
            print(f"Q: {example['query']}\nA: {answer}\nI: {actions}\nE: {example['actions']}")


def test_edit_email_completion():
    for example in email_datasets.EDIT_EMAIL_DATASET_TEST:
        params = example["actions"][0]["params"]
        answer = edit_email_completion(
            params["user_text"], params["first_draft"], params["feedback"]
        )
        print(answer)


def test_summarize_email_completion():
    for example in email_datasets.COMPOSE_EMAIL_DATASET_TEST:
        params = example["actions"][0]["params"]
        answer = summarize_email_completion(params["plain"])
        print(answer)


if __name__ == "__main__":
    # test_send_email_completion()
    # test_search_email_completion()
    test_compose_email_completion()
    test_edit_email_completion()
    test_summarize_email_completion()

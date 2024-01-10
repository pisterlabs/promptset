from curses import raw
import os
import json
import uuid
import time
import subprocess
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    wait_fixed
)  # for exponential backoff

import dashscope
from http import HTTPStatus, client
import openai
from openai import OpenAI

import utils

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI()


@retry(wait=wait_fixed(60), stop=stop_after_attempt(3))
def gpt_completion_with_backoff(**kwargs):
    r"""
    Make a completion request with exponential backoff.
    """
    return client.chat.completions.create(**kwargs)


def gpt_request_turns_with_rate_limiting(model: str, prompt: tuple[int, list[str]], messages: list[dict], seed: int, rate_limit: int = 3, wait_time: int = 60):
    r"""
    Request all turns for a single question. Rate limit the requests.
    """
    raw_responses = []

    for i in range(len(prompt[1])):
        prompt[1][i] = prompt[1][i].replace(
            '\n', '\\n').replace('\r', '\\r')
        messages.append({'role': 'user', 'content': prompt[1][i]})
        # wait. 3 RPM
        if i > 0 and i % rate_limit == 0:
            print("---- sleep for {} seconds".format(wait_time))
            time.sleep(wait_time)

        print("--- turn: ", i+1)

        raw_response = gpt_completion_with_backoff(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=seed
        )
        messages.append({'role': raw_response.choices[0].message.role,
                        'content': raw_response.choices[0].message.content})
        raw_response = utils.convert_gpt_completetion_to_dict(
            raw_response)
        raw_responses.append(raw_response)
    return raw_responses


def gpt_request_questions_with_rate_limiting(model: str, prompts: list[tuple[int, list[str]]], seed: int, task: str, rate_limit: int = 3, wait_time: int = 60):
    raw_responses = {}
    request_count = 0

    # all prompts of the task
    # prompt: tuple[int, list[str]], (question_id, list[question_content_of_turns])
    for prompt in prompts:
        print("- question_id: ", prompt[0])
        question_id = prompt[0]  # int
        language = "zh" if utils.is_chinese(prompt[1]) else "en"
        sys_prompt = get_sys_prompt(language)

        raw_response = []        # list[list[Response]]

        if task == "coding":
            # 3 responses, each: list[Response]
            for i in range(3):
                if request_count > 0 and request_count % rate_limit == 0:
                    print("---- sleep for {} seconds".format(wait_time))
                    time.sleep(wait_time)
                print("-- pass: ", i+1)
                raw_response.append(get_raw_response(
                    sys_prompt, model, language, prompt, seed-1+i, task))
                request_count += 1

        else:
            # 1 response, each: list[Response]
            if request_count > 0 and request_count % rate_limit == 0:
                time.sleep(wait_time)
            print("-- pass: ", 1, "(only 1 pass)")
            raw_response.append(get_raw_response(
                sys_prompt, model, language, prompt, seed, task))
            request_count += 1

        if raw_response:
            # list[Response]
            raw_responses[str(question_id)] = raw_response
        else:
            print("Error: raw response of question {} is empty.".format(question_id))
            return None

    return raw_responses, request_count


def get_sys_prompt(language: str) -> str:
    r"""
    Get the system prompt, depending on the language and task.

    Args: 
        - language: the language of the prompt
    """
    # return "You are a helpful assistant."

    if language == "en":
        return "You are a creative and helpful writing assistant, helping users write article for various purposes. The article topics are very diverse, therefore you must be very creative. If the topic is professional, write in-depth article professionally and accurately. If the topic is creative and entertaining, make it lively and fun, and add some unexpected twists to surprise the readers. Write as detailed as possible."
    if language == "zh":
        return "你是一个富有创意和乐于助人的写作助手，帮助用户撰写各种文章以满足不同的需求。文章的主题非常多样化，因此你必须非常富有创造力。如果主题是专业的，就要以专业和准确的角度来撰写深入的文章。如果主题是创意和娱乐性的，就要让文章生动有趣，加入一些出人意料的转折来惊喜读者。尽可能详细地写作。"
    else:
        raise ValueError("Language must be one of 'en' or 'zh'.")


def get_raw_response(sys_prompt: str, model: str, language: str, prompt: tuple[int, list[str]], seed: int, task: str = "writing"):
    r"""
    Get raw responses of a single question (maybe with multiple turns) for the given task.

    Demand the model to generate a response to the `prompt` for the given task.\
    `prompt` may be single turn or multiple turns, therefore the response may be single turn or multiple turns.\

    `sys_prompt` determined by the task.

    Raw response will be stored in `raw_response_dir` with the name of `task` and `model` and `seed`.

    Args:
        - sys_prompt: the system prompt
        - model: the model name
        - language: the language of the prompt
        - prompt: the prompt, a tuple of (prompt_id, list[prompt_content]), where the list of prompt content is a list of turns
        - seed: the seed of the model
        - task: the task of the response file

    Returns:
       - raw_responses (list[Response]): turns. the raw responses of the conversation, each response is a `Response` object
    """

    # set the system prompt
    # sys_prompt = get_sys_prompt(language)
    messages = [{'role': 'system', 'content': sys_prompt}]

    raw_responses = []

    # qwen-14b-chat, baichuan2-13b-chat-v1
    # no need for rate limitation
    if model == "qwen-14b-chat":
        for i in range(len(prompt[1])):
            print("--- turn: ", i+1)
            prompt[1][i] = prompt[1][i].replace(
                '\n', '\\n').replace('\r', '\\r')
            messages.append({'role': 'user', 'content': prompt[1][i]})
            raw_response = dashscope.Generation.call(
                model=model,
                messages=messages,
                seed=seed,
                top_p=0.8,
                result_format='message',
                enable_search=False,
                max_tokens=1500,
                temperature=1.0,
                repetition_penalty=1.0
            )
            if raw_response.status_code == HTTPStatus.OK:
                messages.append({'role': raw_response.output.choices[0]['message']['role'],
                                'content': raw_response.output.choices[0]['message']['content']})
                raw_responses.append(raw_response)
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    raw_response.request_id, raw_response.status_code,
                    raw_response.code, raw_response.message
                ))
                return None
        return raw_responses

    # baichuan2-13b-chat-v1
    # no need for rate limitation
    if model == "baichuan2-13b-chat-v1":
        for i in range(len(prompt[1])):
            print("--- turn: ", i+1)
            prompt[1][i] = prompt[1][i].replace(
                '\n', '\\n').replace('\r', '\\r')
            messages.append({'role': 'user', 'content': prompt[1][i]})
            raw_response = dashscope.Generation.call(
                model=model,
                messages=messages,
                seed=seed,
                top_p=0.8,
                result_format='message',
                enable_search=False,
                max_tokens=1500,
                temperature=1.0,
                repetition_penalty=1.0
            )
            if raw_response.status_code == HTTPStatus.OK:
                messages.append({'role': raw_response.output.choices[0]['message']['role'],
                                'content': raw_response.output.choices[0]['message']['content']})
                raw_responses.append(raw_response)
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    raw_response.request_id, raw_response.status_code,
                    raw_response.code, raw_response.message
                ))
                return None
        return raw_responses

    # gpt-3.5-turbo
    # need rate limitation
    if model == "gpt-3.5-turbo":
        raw_responses = gpt_request_turns_with_rate_limiting(
            model, prompt, messages, seed, rate_limit=3, wait_time=60)
        return raw_responses


def get_raw_response_of_all_questions_of_task(model: str, prompts: list[tuple[int, list[str]]], raw_response_dir: str, seed: int, task: str = "writing") -> None:
    r"""
    Get raw responses of all question prompts for a task. Write raw responses to a json file.

    3 responses for each prompt, if the task is `"coding"`.

    Raw response will be stored in `raw_response_dir` with the name of `task` and `model` and `seed`.

    Args:
        - model: the model name
        - prompts: the prompts
        - raw_response_dir: the directory to store the raw response file
        - seed: the seed of the model
        - task: task, used to locate the raw response file
    """

    # qwen-14b-chat, baichuan2-13b-chat-v1
    # no need for rate limitation
    if model != "gpt-3.5-turbo":

        raw_responses = {}

        # all prompts of the task
        # prompt: tuple[int, list[str]], (question_id, list[question_content_of_turns])
        for prompt in prompts:
            print("- question_id: ", prompt[0])
            question_id = prompt[0]  # int
            language = "zh" if utils.is_chinese(prompt[1]) else "en"
            sys_prompt = get_sys_prompt(language)

            raw_response = []        # list[list[Response]]
            if task == "coding":
                # 3 responses, each: list[Response]
                for i in range(3):
                    print("-- pass: ", i+1)
                    raw_response.append(get_raw_response(
                        sys_prompt, model, language, prompt, seed-1+i, task))
            else:
                # 1 response, each: list[Response]
                print("-- pass: ", 1, "(only 1 pass)")
                raw_response.append(get_raw_response(
                    sys_prompt, model, language, prompt, seed, task))

            if raw_response:
                # list[Response]
                raw_responses[str(question_id)] = raw_response
            else:
                print("Error: raw response of question {} is empty.".format(
                    question_id))
                return None

    # gpt-3.5-turbo
    # need rate limitation
    else:
        raw_responses, request_count = gpt_request_questions_with_rate_limiting(
            model, prompts, seed, task, rate_limit=3, wait_time=60)
        print(f"Total request count: {request_count}")

    # write raw responses to a json file
    if raw_responses:
        os.makedirs(raw_response_dir, exist_ok=True)
        raw_response_file = os.path.join(
            raw_response_dir, f"{task}_{model}_{seed}.json")
        with open(raw_response_file, "w") as file:
            json.dump({"question_id": raw_responses},
                      file, indent=4, ensure_ascii=False)
            print(
                f"Raw responses of model {model} with seed {seed} of {task} have been written to {raw_response_file}.")


def get_gpt_raw_eval_of_all_questions_of_task(sys_prompt: str, model: str, prompts_for_eval: list[tuple[int, list[str]]], raw_eval_dir: str, seed: int, task: str = "writing", rate_limit: int = 3, wait_time: int = 60) -> None:
    r"""
    Use gpt-3.5-turbo to evaluate the responses of all question prompts for a task. Write raw responses to a json file.

    3 responses for each prompt, if the task is `"coding"`.

    Raw response will be stored in `raw_eval_dir` with the name of `model` and `seed`.

    Args:
        - sys_prompt: the system prompt
        - model: the model name, used to locate the raw eval file
        - prompts_for_eval: the prompts for gpt-3.5-turbo to eval. list[(question_id, [prompt_for_eval])], Inner list means passes
        - raw_eval_dir: the directory to store the raw eval file
        - seed: the seed for eval
        - task: task, used to locate the raw eval file
        - rate_limit: RPM
        - wait_time: sleep time
    """

    # {question_id: list[list[Response]]}, inner list stands for turns (1), outer list stands for passes (1 or 3, 3 for coding)
    raw_evals = {}
    request_count = 0

    # all prompts for eval of the task
    # prompts_for_eval: list[(question_id, [prompt_for_eval])], Inner list means passes, outer list means questions
    for prompt_for_eval in prompts_for_eval:

        # (question_id, [prompt_for_eval_all_pass]) -> [(question_id, prompt_for_eval_single_pass)]
        question_id, promt_for_eval_all_pass = prompt_for_eval
        print("- question_id: ", question_id)
        prompt_for_eval = [(question_id, [prompt_for_eval_single_pass])
                           for prompt_for_eval_single_pass in promt_for_eval_all_pass]
        language = "en"

        # list[list[Response]], inner list stands for turns (1), outer list stands for passes (1 or 3, 3 for coding)
        raw_eval = []

        for i in range(len(prompt_for_eval)):
            if request_count > 0 and request_count % rate_limit == 0:
                print("---- sleep for {} seconds".format(wait_time))
                time.sleep(wait_time)
            raw_eval.append(get_raw_response(sys_prompt, "gpt-3.5-turbo",
                            language, prompt_for_eval[i], seed, task))
            request_count += 1

        if raw_eval:
            raw_evals[str(question_id)] = raw_eval
        else:
            print("Error: raw evaluation of question {} is empty.".format(question_id))
            return None

    # write raw evals to a json file
    if raw_evals:
        os.makedirs(raw_eval_dir, exist_ok=True)
        raw_eval_file = os.path.join(
            raw_eval_dir, f"{task}_{model}_{seed}.json")
        with open(raw_eval_file, "w") as file:
            json.dump({"question_id": raw_evals},
                      file, indent=4, ensure_ascii=False)
            print(
                f"Raw evals of model {model} with seed {seed} of {task} have been written to {raw_eval_file}.")

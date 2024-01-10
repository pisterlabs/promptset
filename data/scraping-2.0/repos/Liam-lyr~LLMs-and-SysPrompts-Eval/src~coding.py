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
        ## starter
        # return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code."
        # return "You are an Python programming assistant who assists the user to write code. Follow the user's requirements carefully and to the letter. First, output the code in a single code block. Secode, describe your plan step-by-step, written out in great detail. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, Make sure that the default file to input (if needed) is 'test_file.txt' in the same directory."
        # file & no input
        ## return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, and that the default file for reading and writing is 'test_file.txt' in the same directory."
        # return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input, and that it has a main function so that it can be ran directly. The default file for reading and writing is 'test_file.txt' in the same directory."
        # return "[Instructions]\n\nYou are an Python programming assistant. Follow the user's requirements carefully to write clean, efficient, and maintainable Python code.\n\nFirst, think step-by-step and describe your plan, written out in great detail.\n\nThen, output the code in a single code block.\n\n[Code Requirements]\n\nYou must use '```python' and '```' to surround the code block.\n\nMake sure that the code does not accept user input.\n\nMake sure that it has a main function so that it can be ran directly.\n\nThe default file for reading and writing is 'test_file.txt' in the same directory."
        # return "You are an Python programming assistant who assists the user to write code. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, Make sure that the default file to input (if needed) is 'test_file.txt' in the same directory."
        return "[Instruction]\n\nYou are an Python programming assistant who assists the user to write code. Follow the user's requirements carefully and to the letter. First, output the code in a single code block. Secode, describe your plan step-by-step, write out in great detail. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input, and that it has a main function so that it can be ran directly, Make sure that the default file to read and write is 'test_file.txt' in the same directory.\n\n[Template]\n\nUser:\n\n[User's question]\n\nAssistant:\n\n```python\n[Generated code]\n```\n\n[Code description]"
    if language == "zh":
        # return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code."
        # return "You are an Python programming assistant who assists the user to write code. Follow the user's requirements carefully and to the letter. First, output the code in a single code block. Secode, describe your plan step-by-step, written out in great detail. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, Make sure that the default file to input (if needed) is 'test_file.txt' in the same directory."
        # return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, and that the default file for reading and writing is 'test_file.txt' in the same directory."
        # return "You are an Python programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input, and that it has a main function so that it can be ran directly, also, the default file for reading and writing is 'test_file.txt' in the same directory."
        # return "[Instructions]\n\nYou are an Python programming assistant. Follow the user's requirements carefully to write clean, efficient, and maintainable Python code.\n\nFirst, think step-by-step and describe your plan, written out in great detail.\n\nThen, output the code in a single code block.\n\n[Code Requirements]\n\nYou must use '```python' and '```' to surround the code block.\n\nMake sure that the code does not accept user input.\n\nMake sure that it has a main function so that it can be ran directly.\n\nThe default file for reading and writing is 'test_file.txt' in the same directory."
        # return "You are an Python programming assistant who assists the user to write code. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan, written out in great detail. Then, output the code in a single code block. Write clean, efficient, and maintainable Python code. Make sure that the code does not accept user input and can be ran directly, Make sure that the default file to input (if needed) is 'test_file.txt' in the same directory."
        return "[Instruction]\n\n您是一个Python编程助手，负责协助用户编写代码。务必要仔细听取用户的要求，并严格按照要求执行。首先，将代码输出在一个代码块中。其次，详细阐述您的计划步骤，写出极为详尽的计划。编写干净、高效、易于维护的Python代码。确保代码不需要接受用户输入，有一个主函数以便可以直接运行。确保默认的读写文件为同目录下的'test_file.txt'。\n\n[Template]\n\n用户:\n\n[用户问题]\n\助手:\n\n```python\n[生成的代码]\n```\n\n[对代码的解释]"
    else:
        raise ValueError("Language must be one of 'en' or 'zh'.")


def get_raw_response(sys_prompt: str, model: str, language: str, prompt: tuple[int, list[str]], seed: int, task: str = "coding"):
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


def get_raw_response_of_all_questions_of_task(model: str, prompts: list[tuple[int, list[str]]], raw_response_dir: str, seed: int, task: str = "coding") -> None:
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


def eval_pass_3(
    code_snippets: list[str],
    code_snippets_storage_root_dir: str,
    question_id: int,
    model: str,
    seed: int,
    verbose_result: bool = False
) -> bool:
    r"""
    Test 3 Python code snippets, generated for the same question prompt, by running them. Test whether at least one success.

    Will save the code snippets as Python files in `code_storage_dir`, then try to run them.

    Args:
        - code_snippets: A list of Python code snippets as str.
        - code_snippets_storage_root_dir: The root directory where the Python files should be saved. Should be `./response/code_snippets`.
        - question_id: The id of the question prompt, used to create code file.
        - verbose_result: Boolean indicating if individual test results should be printed.
        - model: model, used to locate the storage folder
        - seed: seed, used to locate the storage folder

    Returns:
        - Boolean indicating if at least one code snippet ran successfully.
    """

    # Ensure the folder exists
    code_storage_dir = os.path.join(
        code_snippets_storage_root_dir, f"{model}_{seed}")
    os.makedirs(code_storage_dir, exist_ok=True)

    success = False

    for code in code_snippets:
        # Generate a unique file name
        code_file_name = f"code_{question_id}_{uuid.uuid4().hex}.py"
        code_file_path = os.path.join(code_storage_dir, code_file_name)

        try:
            # Write the code to the file in the specified folder
            with open(code_file_path, "w") as file:
                file.write(code)

            # Try to run the Python script
            result = subprocess.run(
                ["python", code_file_name], capture_output=True, text=True, timeout=5, cwd=code_storage_dir)

            # Check if the script ran successfully
            if result.returncode == 0:
                success = True
                if verbose_result:
                    print(f"Code in '{code_file_name}' ran successfully.")

            else:
                if verbose_result:
                    print(
                        f"Code in '{code_file_name}' did not run successfully. Error message:")
                    print(result.stderr)

        except Exception as e:
            print(f"An error occurred while running {code_file_name}: {e}")

    return success


def eval_pass_3_of_all_questions_of_coding(
    # (question_id, list[code_snippets])
    code_snippets_of_all_questions: list[tuple[int, list[str]]],
    code_snippets_storage_root_dir: str,
    model: str,
    seed: int,
    verbose_result: bool = False
) -> list[tuple[int, bool]]:
    r"""
    Test pass@3 metric for coding task.

    Args:
        - code_snippets_of_all_questions: A list of tuples, each tuple is (question_id, list[code_snippets])
        - code_snippets_storage_root_dir: The root directory where the Python files should be saved. Should be `./response/code_snippets`.
        - model: model, used to locate the storage folder
        - seed: seed, used to locate the storage folder
        - verbose_result: Boolean indicating if individual test results should be printed.

    Returns:
        - list[tuple[int, bool]]: list[(question_id, success)]
    """

    results = []

    # test pass@3 for each question
    for i in range(len(code_snippets_of_all_questions)):
        question_id = code_snippets_of_all_questions[i][0]      # int
        code_snippets = code_snippets_of_all_questions[i][1]    # list[str]
        success = eval_pass_3(
            code_snippets=code_snippets,
            code_snippets_storage_root_dir=code_snippets_storage_root_dir,
            question_id=question_id,
            model=model,
            seed=seed,
            verbose_result=verbose_result
        )

        results.append((question_id, success))
        if success:
            print(
                f"At least one code snippet ran successfully for question {question_id}.")
        else:
            print(
                f"No code snippet ran successfully for question {question_id}.")

    return results


def get_gpt_raw_eval_of_all_questions_of_task(sys_prompt: str, model: str, prompts_for_eval: list[tuple[int, list[str]]], raw_eval_dir: str, seed: int, task: str = "coding", rate_limit: int = 3, wait_time: int = 60) -> None:
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

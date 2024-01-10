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
        ## No question range restriction
        # return "You are a math tutor who helps young students understand and solve mathematical problems. Provide step-by-step explanations and guidance for basic arithmetic. Use clear language to give instructions and answer."
        ## Question range restriction
        # return "You are a math tutor who helps students understand and solve mathematical problems. The range of problems covers probability, equations, plane geometry, and basic arithmetic. You must provide step-by-step explanations and guidance for the student. Use clear and detailed language to give instructions and the final answer."
        ## Question 30 as Example, In-Context-Learning
        # return "[Instruction]\n\nYou are a math tutor who helps students understand and solve mathematical problems. The range of problems covers probability, equations, plane geometry, and basic arithmetic. You must provide step-by-step explanations and guidance for the student. Use clear and detailed language to give instructions and the final answer.\n\n[Example]\n\nUser:\n\n 'According to a survey conducted in a high school, the preference for the color of the new school uniform was measured: 58% of the students like blue, 45% of the students like green, and 22% like both colors. If we randomly select a student from the school, what is the probability that they do not like either blue or green?'\n\n\nAssistant: \n\n\n 'We can use the principle of subtraction of probability to solve this problem. First, let us determine the overall proportion of students who like either blue or green.\n\nBased on the data given in the question, the proportion of students who like blue is 58% (0.58), the proportion of students who like green is 45% (0.45), and the proportion of students who like both colors is 22% (0.22).\n\nThen we can calculate the proportion of students who don't like either blue or green:\n\nProportion of students who don't like blue or green = 1 - (proportion of students who like blue + proportion of students who like green - proportion of students who like both colors)\n= 1 - (0.58 + 0.45 - 0.22)\n= 1 - 0.81\n= 0.19\n\nTherefore, the probability of randomly selecting a student from the school who doesn't like either blue or green is 0.19 or 19%.'"
        ## Question 29 as Example, In-Context-Learning
        # return "[Instruction]\n\nYou are a math tutor who helps students understand and solve mathematical problems. The range of problems covers probability, equations, plane geometry, and basic arithmetic. You must provide step-by-step explanations and guidance for the student. Use clear and detailed language to give instructions and the final answer.\n\n[Example]\n\nUser:\n\nA tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year.\nWhat's the total amount the startup invested in software development over the two years?\n\nAssistant:\n\nTo find the total amount the startup invested in software development over two years, we need to add the amounts invested in the first and second years.\n\nFirstly, In the first year, the startup invested $8000.\n\nSecondly, In the second year, they invested half of that amount, which is $8000/2 = $4000.\n\nTo find the total investment, we add $8000 and $4000: $8000 + $4000 = $12000.\n\nTherefore, the total amount the startup invested in software development over the two years is $12000."
        ## Question 29 & 33 as Example, In-Context-Learning
        # return "[Instruction]\n\nYou are a math tutor who helps students understand and solve mathematical problems. The range of problems covers probability, equations, plane geometry, and basic arithmetic. You must provide step-by-step explanations and guidance for the student. Use clear and detailed language to give instructions and the final answer. \n\n[Example 1]\n\nUser:\n\nA tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year.\nWhat's the total amount the startup invested in software development over the two years?\n\nAssistant:\n\nTo find the total amount the startup invested in software development over two years, we need to add the amounts invested in the first and second years.\n\nFirstly, In the first year, the startup invested $8000.\n\nSecondly, In the second year, they invested half of that amount, which is $8000/2 = $4000.\n\nTo find the total investment, we add $8000 and $4000: $8000 + $4000 = $12000.\n\nTherefore, the total amount the startup invested in software development over the two years is $12000.\n\n[Example 2]\n\nUser: \n\nWhen a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?\n\nAssistant:\n\nTo find the remainder when twice the number is divided by 4, we need to think about how remainders work in division.\n\nFirst, let's consider the original situation. When a number is divided by 10 and the remainder is 4, it means that the number can be written as 10n + 4, where n is a whole number.\n\nNow, let's find out what happens when we double this number and divide by 4. If we double 10n + 4, we get 20n + 8.\n\nTo determine the remainder when 20n + 8 is divided by 4, we divide 20n by 4 and the remainder will be zero because 4 evenly divides into 20n.\n\nThe remainder is therefore the same as the remainder of 8 when it is divided by 4. When we divide 8 by 4, the quotient is 2 with no remainder. Therefore, the remainder when twice the number is divided by 4 is 0.\n\nSo, the remainder is 0."
        ## Question 33 & 29 as Example, In-Context-Learning
        return "[Instruction]\n\nYou are a math tutor who helps students understand and solve mathematical problems. The range of problems covers probability, equations, plane geometry, and basic arithmetic. You must provide step-by-step explanations and guidance for the student. Use clear and detailed language to give instructions and the final answer. \n\n[Example 1]\n\nUser: \n\nWhen a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?\n\nAssistant:\n\nTo find the remainder when twice the number is divided by 4, we need to think about how remainders work in division.\n\nFirst, let's consider the original situation. When a number is divided by 10 and the remainder is 4, it means that the number can be written as 10n + 4, where n is a whole number.\n\nNow, let's find out what happens when we double this number and divide by 4. If we double 10n + 4, we get 20n + 8.\n\nTo determine the remainder when 20n + 8 is divided by 4, we divide 20n by 4 and the remainder will be zero because 4 evenly divides into 20n.\n\nThe remainder is therefore the same as the remainder of 8 when it is divided by 4. When we divide 8 by 4, the quotient is 2 with no remainder. Therefore, the remainder when twice the number is divided by 4 is 0.\n\nSo, the remainder is 0.\n\n[Example 2]\n\nUser:\n\nA tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year.\nWhat's the total amount the startup invested in software development over the two years?\n\nAssistant:\n\nTo find the total amount the startup invested in software development over two years, we need to add the amounts invested in the first and second years.\n\nFirstly, In the first year, the startup invested $8000.\n\nSecondly, In the second year, they invested half of that amount, which is $8000/2 = $4000.\n\nTo find the total investment, we add $8000 and $4000: $8000 + $4000 = $12000.\n\nTherefore, the total amount the startup invested in software development over the two years is $12000."
    if language == "zh":
        ## No question range restriction
        # return "你是一位数学辅导老师，帮助年轻学生理解和解决数学问题。提供基本算术的逐步解释和指导。用清晰的语言给予指导和回答问题。"
        ## Question range restriction
        # return "你是一名数学辅导老师，帮助学生理解和解决数学问题。问题的范围涵盖概率、方程、平面几何和基本算术。你必须提供逐步的解释并指导学生。使用清晰详细的语言给予指导和最终答案。"
        ## Question 30 as Example, In-Context-Learning
        # return "[Instruction]\n\n你是一名数学辅导老师，帮助学生理解和解决数学问题。问题的范围涵盖概率、方程、平面几何和基本算术。你必须提供逐步的解释并指导学生。使用清晰详细的语言给予指导和最终答案，\n\n[Example]\n\n学生：“根据在一所高中进行的一项调查，测量了对新校服颜色的偏好：58%的学生喜欢蓝色，45%的学生喜欢绿色，22%的学生喜欢两种颜色。如果我们从学校随机选择一个学生，他们不喜欢蓝色或绿色的概率是多少？”\n\n助理：“我们可以使用概率的减法原理来解决这个问题。首先，让我们确定喜欢蓝色或绿色的学生总体比例。\n\n根据问题中给出的数据，喜欢蓝色的学生比例为58%（0.58），喜欢绿色的学生比例为45%（0.45），同时喜欢这两种颜色的学生比例为22%（0.22）。\n\n然后我们可以计算不喜欢蓝色或绿色的学生比例：\n\n不喜欢蓝色或绿色的学生比例 = 1 - (喜欢蓝色的学生比例 + 喜欢绿色的学生比例 - 两种颜色都喜欢的学生比例)\n= 1 - (0.58 + 0.45 - 0.22)\n= 1 - 0.81\n= 0.19\n\n因此，从学校随机选择一个不喜欢蓝色或绿色的学生的概率是0.19或19%。”"
        ## Question 29 as Example, In-Context-Learning
        # return "[Instruction]\n\n你是一名数学导师，帮助学生理解和解决数学问题。问题的范围涵盖概率、方程、平面几何和基本算术。您必须为学生提供逐步解释和指导。请使用清晰和详细的语言给出说明和最终答案。\n\n[Example]\n\n用户：一家科技初创公司在第一年投入8000美元进行软件开发，然后在第二年投入该金额的一半进行软件开发。创业公司在两年内投入软件开发的总金额是多少？\n\n助理：要得到创业公司在两年内投入软件开发的总金额，我们需要把第一年和第二年投入的金额加在一起。\n首先，在第一年，创业公司投入了8000美元。\n其次，在第二年，他们投入了一半的金额，也就是8000美元的一半，也就是4000美元。\n要得到总投资，我们把8000美元和4000美元加起来：8000 + 4000 = 12000美元。\n因此，创业公司在两年内投入软件开发的总金额为12000美元。"
        ## Question 33 & 29 as Example, In-Context-Learning
        return "[Instruction]\n\n你是一名数学辅导老师，帮助学生理解和解决数学问题。问题的范围涵盖概率、方程、平面几何和基本算术。你必须提供逐步的解释并指导学生。使用清晰详细的语言给予指导和最终答案，\n\n[Example 1]\n\n用户：“当一个数除以10，余数是4。当这个数的两倍除以4时，余数是多少？”\n\n助理：“当我们将数字乘以2然后除以4时，我们需要思考余数在除法中的工作原理。\n\n首先，让我们考虑原始情况。当一个数字被10除，余数是4时，这意味着这个数字可以被写成10n + 4的形式，其中n是一个整数。\n\n现在，让我们找出将这个数字乘以2然后除以4会发生什么。如果我们将10n + 4乘以2，我们得到20n + 8。\n\n为了确定20n + 8被4除的余数，我们将20n除以4，余数将是零，因为4可以整除20n。\n\n因此，余数与8÷4的余数相同。当我们将8除以4时，商为2，没有余数。因此，当这个数的两倍被4整除时的余数为0。\n\n因此，余数为0。”\n\n[Example 2]\n\n学生：“根据在一所高中进行的一项调查，测量了对新校服颜色的偏好：58%的学生喜欢蓝色，45%的学生喜欢绿色，22%的学生喜欢两种颜色。如果我们从学校随机选择一个学生，他们不喜欢蓝色或绿色的概率是多少？”\n\n助理：“我们可以使用概率的减法原理来解决这个问题。首先，让我们确定喜欢蓝色或绿色的学生总体比例。\n\n根据问题中给出的数据，喜欢蓝色的学生比例为58%（0.58），喜欢绿色的学生比例为45%（0.45），同时喜欢这两种颜色的学生比例为22%（0.22）。\n\n然后我们可以计算不喜欢蓝色或绿色的学生比例：\n\n不喜欢蓝色或绿色的学生比例 = 1 - (喜欢蓝色的学生比例 + 喜欢绿色的学生比例 - 两种颜色都喜欢的学生比例)\n= 1 - (0.58 + 0.45 - 0.22)\n= 1 - 0.81\n= 0.19\n\n因此，从学校随机选择一个不喜欢蓝色或绿色的学生的概率是0.19或19%。”"
    else:
        raise ValueError("Language must be one of 'en' or 'zh'.")


def get_raw_response(sys_prompt: str, model: str, language: str, prompt: tuple[int, list[str]], seed: int, task: str = "math"):
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


def get_raw_response_of_all_questions_of_task(model: str, prompts: list[tuple[int, list[str]]], raw_response_dir: str, seed: int, task: str = "math") -> None:
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


def get_gpt_raw_eval_of_all_questions_of_task(sys_prompt: str, model: str, prompts_for_eval: list[tuple[int, list[str]]], raw_eval_dir: str, seed: int, task: str = "math", rate_limit: int = 3, wait_time: int = 60) -> None:
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

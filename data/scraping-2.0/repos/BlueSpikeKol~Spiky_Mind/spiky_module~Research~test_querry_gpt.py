import os
import json
import inspect

import openai

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4"
openai.api_key = OPENAI_KEY
ENCODING = "utf-8"


def request_first_message(msg: str) -> dict:
    """
    First test function to try to get the question from chatGPT depending on the project you have
    :param msg: Ty pe of project you have
    :return: The answer from chatGPT
    """
    msg = f"Give me all the relevant topic of information I should ask the user for the following project.\n" + msg

    messages = [{"role": "user", "content": msg}]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
    )
    response_message = response["choices"][0]["message"]

    print(response_message)
    print(response_message["content"])
    save_test_information(msg, response_message, inspect.stack()[0][3])

    return response_message


def format_msg_1(msg: str) -> str:
    return (f"Give few short questions to get more informations about the following project. "
            f"The information you want to know should be useful to organize the project and have a better "
            f"understanding of it. The project: \n") + msg


def format_msg_2(msg: str) -> str:
    return f"Give point to complete in a form to gather quantitative information about the following project: \n" + msg


def format_msg_3(msg: str) -> str:
    return (f"Give question to create a google form type of form. Those questions should aim to gather information"
            f" about the following project. You only want to gather information useful to organize the project and"
            f" help people realizing the project. The project is the following: \n") + msg


def format_msg_4(msg: str) -> str:
    return (f"Give question to create a google form type of form. Those questions should aim to gather quantitative"
            f" information about the following project. The project:\n") + msg


def format_msg_5(msg: str) -> str:
    return (f"Give points to gather quantitative information about the following project.You want to extract general"
            f" informations to organize the building of the project."
            f" The project:\n") + msg


def format_msg_6(msg: str) -> str:
    return (f"Give points to gather quantitative information about the following project.You want to extract general"
            f" informations to organize the building of the project. Do not include the answer of your points."
            f" The project:\n") + msg


def format_msg_7(msg: str) -> str:
    return (f"Give points to gather quantitative information about the following project.You want to extract general"
            f" informations to organize the building of the project. Your point should ba as follows:"
            f"<statement>: < <answer>. Do not include the answer."
            f" The project:\n") + msg


def format_msg_8(msg: str) -> str:
    return (f"Ask question to gather quantitative information only about the following project. The format "
            f"of your questions should be: <question> <answer>. Do not include the answer.") + msg


# For now the best format for generating the form (asking only quantitative questions)
def format_msg_9(msg: str) -> str:
    return (f"Ask question to gather only quantitative information about the following project. "
            f"Make sure your question are simple and general. The format "
            f"of your questions should be: <question> <answer>. Do not include the answer.") + msg


def request_message(msg: str, fn_format) -> dict:
    msg = fn_format(msg)

    messages = [{"role": "user", "content": msg}]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
    )
    response_message = response["choices"][0]["message"]

    save_test_information(msg, response_message, fn_format.__name__)

    return response_message


def save_test_information(prompt: str, answer: str, fn_name: str) -> None:
    """
    Save the given information into a JSON file with the index of the data

    :param prompt: The prompt given to chatGPT
    :param answer: The answer returned by chatGPT
    :param fn_name: The name of the function used to make the discussion with chatGPT
    """

    with open("nb_test.txt", "r") as fic:
        nb = int(fic.readlines()[0])

    with open("nb_test.txt", "w") as fic:
        fic.write(str(nb + 1))

    with open("result_tests.json", "r", encoding=ENCODING) as fic:
        data = json.load(fic)

    data[f"test_{nb}"] = {"function": fn_name, "model": MODEL, "prompt": prompt}

    ls_answer = answer["content"].split("\n")

    index = 0

    for i in ls_answer:
        if len(i) == 0:
            continue
        else:
            data[f"test_{nb}"][f"Q{index}"] = i
            index += 1

    with open("result_tests.json", "w", encoding=ENCODING) as fic:
        json.dump(data, fic, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    project_description = (f"The High School Coding Club in Anytown, USA, aims to bridge this knowledge gap with a "
                           f"unique project targeted at elementary school children. The goal? To create an interactive,"
                           f" web-based platform that introduces basic coding concepts through gamified experiences,"
                           f" making technology accessible and enjoyable for the youngest learners.")

    # request_first_message(project_description)
    request_message(project_description, format_msg_9)

    print(save_test_information.__name__)
    print(save_test_information)
    print(type(save_test_information))

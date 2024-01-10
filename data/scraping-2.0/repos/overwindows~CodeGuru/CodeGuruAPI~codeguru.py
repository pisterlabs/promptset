import os
import openai
import argparse
from prompts import *

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_testcases(input: str) -> str:
    prompt = get_prompt('test_generation')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def fix_bug(input: str) -> str:
    prompt = get_prompt('bug_fix')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def optimize_perf(input: str) -> str:
    prompt = get_prompt('perf_optimization')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def summarize_code(input: str) -> str:
    prompt = get_prompt('function_description')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def beautify_code(input: str) -> str:
    prompt = get_prompt('beautify')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def review_code(input: str) -> str:
    prompt = get_prompt('code_review')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def transpile_code(input: str) -> str:
    prompt = get_prompt('code_translation')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def commit_msg(input: str) -> str:
    prompt = get_prompt('commit_message')
    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)
    return answer

def get_prompt(task: str) -> str:
    if task == "code_review":
        return PROMPT_CODE_REVIEW
    elif task == "commit_message":
        return PROMPT_COMMIT_MESSAGE
    elif task == "variable_name":
        return PROMPT_VARIABLE_NAME
    elif task == "function_signature":
        return PROMPT_FUNCTION_SIGNATURE
    elif task == "function_description":
        return PROMPT_FUNCTION_DESCRIPTION
    elif task == "perf_optimization":
        return PROMPT_PERF_OPTIMIZATION
    elif task == "bug_fix":
        return PROMPT_BUG_FIX
    elif task == "test_generation":
        return PROMPT_TEST_GENERATION
    elif task == "code_translation":
        return PROMPT_CODE_TRANSLATION
    elif task == "beautify":
        return PROMPT_BEAUTIFY
    else:
        raise ValueError("Invalid task: " + task)


def ask_question(question: str) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        temperature=0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
    )
    return response.choices[0].text


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="CodeGuru: A tool to generate code from natural language prompts")
    argparser.add_argument("--task", type=str, default="code_review",
                        choices=["code_review", "commit_message", "variable_name", "function_signature", "function_description", "perf_optimization", "bug_fix", "test_generation", "code_translation"])
    args = argparser.parse_args()

    # prompt = "\"\"\"\nUtil exposes the following:\nutil.openai() -> authenticates & returns the openai module, which has the following functions:\nopenai.Completion.create(\n    prompt=\"<my prompt>\", # The prompt to start completing from\n    max_tokens=123, # The max number of tokens to generate\n    temperature=1.0 # A measure of randomness\n    echo=True, # Whether to return the prompt in addition to the generated completion\n)\n\"\"\"\nimport util\n\"\"\"\nCreate an OpenAI completion starting from the prompt \"Once upon an AI\", no more than 5 tokens. Does not include the prompt.\n\"\"\"\n"
    prompt = get_prompt(args.task)
    # print(prompt)
    input = None
    with open("codeguru.in", "r") as f:
        input = f.read()

    question = prompt + "\n\n" + input + "\"\"\""
    answer = ask_question(question)

    with open("codeguru.out", "w") as f:
        f.write(answer)

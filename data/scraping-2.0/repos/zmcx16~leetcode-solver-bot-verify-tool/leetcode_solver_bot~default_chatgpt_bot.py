import argparse
import json
import logging
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


OPENAI_API_KEY = ""
MODEL_NAME = "gpt-3.5-turbo-16k-0613"

SYSTEM_ROLE_PROMPT = """You are a professional coding robot that generates answers based on programming questions.
You will maintain professionalism and aim to make the code as simple, fast, accurate, and efficient as possible.
You will write comment on your code to tell user why you write the line
The solution code should primarily contain only and begin with necessary Python3 built-in libraries.
Followed by the actual Python3 code.
Ensure the code does NOT contain any decorative or non-functional text or testing code.
Perform a virtual 'code review' before finalizing your response to ensure correctness.
You will read the meaning of the question carefully before answering,and base on the question meaning to generate answer.
"""

HUMAN_ROLE_PROMPT = """"Here is the programming requirement you need to solve:
{problem}
----------- program_requirement end -----------
Important : You should output a JSON string that includes the fields 'thought' and 'solution_code', and they should both be of string type.
"""


def solve_problem(problem, openai_api_key, model_name, temperature):
    if not openai_api_key:
        raise ValueError("openai_api_key is required. Please provide it as an argument")
    if not model_name:
        raise ValueError(
            "model_name is required. Please provide it as an argument")

    sp_bot = ChatOpenAI(
        model_name=model_name,
        openai_api_key=openai_api_key,
        temperature=temperature)

    prompt = PromptTemplate(template=HUMAN_ROLE_PROMPT, input_variables=["problem"], )
    messages = [SystemMessage(content=SYSTEM_ROLE_PROMPT),
                HumanMessage(content=prompt.format_prompt(problem=problem).to_string())]

    response = sp_bot(messages).content
    logging.info("ChatGPT response:\n" + response)
    response_json = json.loads(response)
    return response_json["solution_code"]


if __name__ == "__main__":
    # record time
    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "-log-level", dest="log_level", default="INFO")
    parser.add_argument("-i", "-problem-path", dest="problem_path", default="./problem.txt")
    parser.add_argument("-o", "-solution-path", dest="solution_path", default="./solution.py")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    try:
        with open(args.problem_path, "r", encoding="utf-8") as file:
            problem = file.read()

        solution_code = solve_problem(problem, OPENAI_API_KEY, MODEL_NAME, 0.7)
        with open(args.solution_path, "w") as f:
            f.write(solution_code)
    except Exception as e:
        logging.error(f"Exception:\n {str(e)} happened!", exc_info=True)

    # record time
    end = datetime.now()
    logging.info(f"Total time used: {end - now}")

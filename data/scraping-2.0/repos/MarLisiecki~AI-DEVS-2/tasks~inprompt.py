import re

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.solver import Solver
from tasks.blogger import prepare_prompt

ASSISTANT_CONTENT = "Based on this information: {context}, answer to the question"
USER_CONTENT = "Question: {question}"


def inprompt(input_data: dict) -> dict:
    input_list = input_data["input"]
    question = input_data["question"]
    name = re.search(r"\b[A-Z][a-z]*\b", question).group()
    filtered_list = [sentence for sentence in input_list if name in sentence]
    oai = OpenAIConnector()
    prompt = prepare_prompt(
        ASSISTANT_CONTENT.format(context="".join(filtered_list)),
        USER_CONTENT.format(question=question),
    )
    verification_result = oai.generate_answer(
        model=OpenAiModels.gpt3_5_turbo.value, messages=prompt
    )
    prepared_answer = {"answer": verification_result}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("inprompt")
    sol.solve(inprompt)

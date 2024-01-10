from pprint import pprint

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.solver import Solver
from tasks.blogger import prepare_prompt

ASSISTANT_CONTENT = "You need to verify the answer to a question. If the answer is correct return YES, if not return NO"
USER_CONTENT = "Question: {question}. Answer: {answer}"


def liar(question: str, answer_from_api: dict):
    answer = answer_from_api["answer"]
    oai = OpenAIConnector()
    prompt = prepare_prompt(
        ASSISTANT_CONTENT, USER_CONTENT.format(question=question, answer=answer)
    )
    verification_result = oai.generate_answer(
        model=OpenAiModels.gpt3_5_turbo.value, messages=prompt
    )
    prepared_answer = {"answer": verification_result}
    return prepared_answer


if __name__ == "__main__":
    question = "Is overment a bio-robot?"
    sol = Solver("liar")
    sol.solve(liar, question=question)

from typing import List

from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.solver import Solver


def moderation(input_data: dict):
    sentences: List[str] = input_data["input"]
    oai = OpenAIConnector()
    verdicts = [*map(int, oai.moderate_prompt(sentences))]
    prepared_answer = {"answer": verdicts}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("moderation")
    sol.solve(moderation)

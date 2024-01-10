import json

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from utils.common_utils import get_azure_chatbot
from utils.log import logger

QUESTION_SOLVER_PROMPT_PATH = r"./prompt/question_solver_prompt.txt"


class QuestionSolver:
    def __init__(self) -> None:
        self.chatbot = get_azure_chatbot(request_timeout=20)
        with open(QUESTION_SOLVER_PROMPT_PATH, "r") as file:
            question_solver_prompt = file.read()
        self.prompt_template = question_solver_prompt

    def solve(self, program_requirement: str) -> dict:
        """
        This function accepts a programming requirement as an input, and uses a chatbot to generate a solution.
        The solution is then parsed and returned in a dictionary format.

        Parameters:
        program_requirement (str): A string that describes the programming requirement.

        Returns:
        result (dict): A dictionary containing two keys - 'thought' and 'solution_code'.
            The 'thought' key contains the thought process behind the solution,
            while the 'solution_code' key contains the actual code for the solution.

        Raises:
        ValueError: If the response from ChatGPT is in the wrong format, the function will raise a ValueError.

        """
        prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["program_requirement"],
        )
        input = prompt.format_prompt(program_requirement=program_requirement)
        logger.debug("Input prompt\n" + input.to_string())

        messages = [HumanMessage(content=input.to_string())]
        response = self.chatbot(messages).content
        logger.debug("ChatGPT response:\n" + str(response))

        try:
            result = json.loads(response)
            logger.debug("Parsed result:\n" + str(result))
        except Exception:
            raise ValueError(
                "ChatGPT's response is in wrong format, please try again or adjust the prompt."
            )
        return result

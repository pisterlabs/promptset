import os
import sys
from typing import *

from util import remove_indent
from core.custom_callback_handler import CustomCallbackHandler
from langchain.chat_models import ChatOpenAI


class DataManager:
    def __init__(self):
        self.company = None
        self.job = None
        self.requirement = None
        self.coverletter = None
        self.introduce = None
        self.userdata = None

    def set_data(self, input_data: Dict):
        self.company = input_data["user_company"]
        self.job = input_data["user_job"]
        self.requirement = input_data["job_requirement"]
        self.coverletter = input_data["cover_letter"]
        self.introduce = input_data["self_introduce"]
        self.userdata = remove_indent(
            f"""JobAdvertisement At {self.company}:
            {self.requirement}
            
            Interviewee`s desired position:
            {self.job}
            
            Interviewee`s coverletter:
            {self.coverletter}
            
            Interviewee`s self-introduction:
            {self.introduce}
            """)

    def get_data(self) -> Dict:
        return {
            "company": self.company,
            "job": self.job,
            "requirement": self.requirement,
            "coverletter": self.coverletter,
            "introduce": self.introduce
        }

    def get_userdata(self) -> str:
        return self.userdata


class KeyManager:
    def __init__(self):
        self.openai_api_key = None
        self.serpapi_api_key = None
        if sys.platform == 'darwin':
            # MacOS
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
            self.serpapi_api_key = os.environ["SERPAPI_API_KEY"]
        elif sys.platform == 'win32':
            # Windows
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")


class ChatManager:
    def __init__(
        self,
        custom_callback_handler: Type = CustomCallbackHandler,
        streaming: bool = True,
        model: str = "gpt-3.5-turbo",
        temperature: int = 0.5
    ):
        """

        :param custom_callback_handler: CustomCallbackHandler
        :param streaming: Streaming
        :param model: GPT Model
        :param temperature: Temperature
        """
        self.chat = ChatOpenAI(
            openai_api_key=KeyManager().openai_api_key,
            streaming=streaming,
            callbacks=[custom_callback_handler()],
            model_name=model,
            temperature=temperature,
        )

    def get_chat_model(self):
        return self.chat


class QuestionManager:
    def __init__(self, question_list: List):
        self.question = [] + question_list
        self.number = len(self.question)
        self.count = -1
        self.max_number = 6

    def set_max_question_number(self, number: int):
        if number > self.number:
            self.max_number = self.number
        self.max_number = number

    def get_question(self) -> str:
        self.count += 1
        if self.count > self.max_number:
            return "No questions."
        return self.question[self.count]

    def ask_question_count(self) -> int:
        return self.count + 1


class QuestionEntity:
    def __init__(self, question: str):
        self.question = question
        self.answer = None

    def add_answer(self, answer: str):
        self.answer = answer


class EvaluationManager:
    def __init__(self):
        self.evaluation_records = {
            "coverletter": None,
            "answer": []
        }

    def add_coverletter_evaluation(self, evaluation: str):
        self.evaluation_records["coverletter"] = evaluation

    def add_answer_evaluation(self, evaluation: str):
        self.evaluation_records["answer"].append(evaluation)

    def get_answer_evaluation(self) -> str:
        return self.evaluation_records["answer"][-1]

    def get_all_evaluation(self) -> Dict:
        return self.evaluation_records

import openai
from typing import Dict
from user import User
from immigration_data import ImmigrationData

class AIAgent:
    """Class to create an AI agent that answers New Zealand immigration questions and creates immigration plans."""

    def __init__(self, model: str = "text-davinci-002", data: ImmigrationData = ImmigrationData()):
        """Initializes the AIAgent class with the GPT-3 model and the immigration data."""
        self.model = model
        self.data = data

    def answer_question(self, question: str) -> str:
        """Uses the GPT-3 model to answer a given question."""
        response = openai.Completion.create(engine=self.model, prompt=question, max_tokens=150)
        return response.choices[0].text.strip()

    def create_plan(self, user: User) -> Dict[str, str]:
        """Creates a personalized immigration plan for a given user."""
        plan = {}
        plan["name"] = user.name
        plan["email"] = user.email
        plan["immigration_status"] = user.immigration_status
        plan["immigration_goal"] = user.immigration_goal
        plan["plan"] = self.answer_question(f"How can someone with {user.immigration_status} status achieve their goal of {user.immigration_goal} in New Zealand?")
        return plan

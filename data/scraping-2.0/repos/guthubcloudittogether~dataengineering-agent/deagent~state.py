"""Welcome to Reflex! This app is a demonstration of OpenAI's GPT."""
import reflex as rx
from .helpers import navbar
import openai
import datetime
from deagent import style
from deagent.conversation import Conversation
from typing import Optional
from deagent.openaichat_util import chat_completion_with_function_execution
from deagent.functions import *

openai.api_key = "sk    F "
MAX_QUESTIONS = 10


class User(rx.Model, table=True):
    """A table for users in the database."""

    username: str
    password: str


class Question(rx.Model, table=True):
    """A table for questions and answers in the database."""

    username: str
    prompt: str
    answer: str
    timestamp: datetime.datetime = datetime.datetime.now()


class State(rx.State):
    """The app state."""

    show_columns = ["Question", "Answer"]
    username: str = ""
    password: str = ""
    logged_in: bool = False

    prompt: str = ""
    result: str = ""
    status: str = ""
    metastatus: str = ""

    def __init__(self):
        self.__init__(None)

    def __init__(self, hn_conversation: Conversation):
        self.hn_conversation = hn_conversation

    @rx.var
    def questions(self) -> list[Question]:
        """Get the users saved questions and answers from the database."""
        with rx.session() as session:
            if self.logged_in:
                qa = (
                    session.query(Question)
                    .where(Question.username == self.username)
                    .distinct(Question.prompt)
                    .order_by(Question.timestamp.desc())
                    .limit(MAX_QUESTIONS)
                    .all()
                )
                return [[q.prompt, q.answer] for q in qa]
            else:
                return []

    def get_result(self):
        if (
            rx.session()
            .query(Question)
            .where(Question.username == self.username)
            .where(Question.prompt == self.prompt)
            .first()
            or rx.session()
            .query(Question)
            .where(Question.username == self.username)
            .where(
                Question.timestamp
                > datetime.datetime.now() - datetime.timedelta(days=1)
            )
            .count()
            > MAX_QUESTIONS
        ):
            return rx.window_alert(
                "You have already asked this question or have asked too many questions in the past 24 hours."
            )
        try:
            response = openai.Completion.create(
                model="gpt-4-0613",
                prompt=self.prompt,
                temperature=0,
                max_tokens=100,
                # top_p=1,
            )
            self.result = response["choices"][0]["text"].replace("\n", "")
        except:
            return rx.window_alert("Error occured with OpenAI execution.")

    def ask_hn(self):
        print(f"PROMPT ****** {self.prompt}")
        self.hn_conversation.add_message("user", self.prompt)
        self.status = "Processing Query"

        chat_response = chat_completion_with_function_execution(
            self.hn_conversation.conversation_history, functions=hnapi_functions
        )
        try:
            self.result = chat_response
            print(f" RESULY ** {self.result}")
            # hn_conversation.add_message("assistant", self.result)
        except Exception as e:
            print(e)
            return rx.window_alert("Error occured with OpenAI execution.")

    def save_result(self):
        with rx.session() as session:
            answer = Question(
                username=self.username, prompt=self.prompt, answer=self.result
            )
            session.add(answer)
            session.commit()

    def set_username(self, username):
        self.username = username.strip()

    def set_password(self, password):
        self.password = password.strip()

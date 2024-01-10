"""Welcome to Reflex! This app is a demonstration of OpenAI's GPT."""
import datetime
import os

import openai
import reflex as rx
from dotenv import load_dotenv

from deagent import style
from deagent.conversation import Conversation
from deagent.functions import *
from deagent.openaichat_util import chat_completion_with_function_execution
from .helpers import navbar

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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

    prompt: str = ""
    result: str = ""
    status: str = "DE Hackernews Agent"
    metastatus: str = "OpenAI Function Calling Demo"
    is_uploading: bool

    def ask_hn(self):
        self.is_uploading = True
        print(f"PROMPT ****** {self.prompt}")
        hn_conversation.add_message("user", self.prompt)
        self.status = "Processing Query"
        self.metastatus = "Extracting information from hackernews"

        chat_response = chat_completion_with_function_execution(
            hn_conversation.conversation_history, functions=hnapi_functions
        )
        self.is_uploading = False
        try:
            if chat_response:
                self.status = "Response Received"
                self.metastatus = f"Total Response length {len(chat_response)}"
            else:
                self.status = "No Response"
                self.metastatus = ""
            self.result = chat_response
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


def status():
    return rx.center(
        rx.vstack(
            rx.alert(
                rx.alert_icon(),
                rx.alert_title(State.status),
                rx.alert_description(State.metastatus),
                status="success",
                variant="subtle",
            ),
            border="1px solid #eaeaef",
            padding="2rem",
            border_radius=8,
            margin_left="10rem",
            # align_items="right",
            # overflow="right"
        )
    )


def index():
    return rx.center(
        navbar(State),
        rx.vstack(
            rx.center(
                rx.vstack(
                    rx.cond(
                        State.is_uploading,
                        rx.progress(is_indeterminate=True, color="blue", width="100%"),
                        rx.progress(value=0, width="100%"),
                    ),
                    rx.text_area(
                        default_value=State.result,
                        placeholder="HN Result",
                        is_disabled=State.is_uploading,
                        width="100%",
                        height="90%",
                        is_read_only=True,
                    ),
                    shadow="lg",
                    padding="1em",
                    border_radius="lg",
                    width="100%",
                    height="400px",
                ),
                width="100%",
            ),
            rx.center(
                rx.vstack(
                    rx.hstack(
                        rx.input(
                            placeholder="Ask a question",
                            is_disabled=State.is_uploading,
                            style=style.input_style,
                            on_blur=State.set_prompt,
                            width="100%",
                        ),
                        rx.button(
                            "Ask", style=style.button_style, on_click=State.ask_hn
                        ),
                        width="500px",
                    )
                ),
                shadow="lg",
                padding="2em",
                border_radius="lg",
                width="100%",
            ),
            status(),
            width="80%",
            height="80%",
            spacing="2em",
        ),
        padding_top="6em",
        text_align="top",
        position="relative",
    )


hn_system_message = """You are a DataEngineering Agent, a helpful assistant reads hackernews to answer user questions.
You summarize the hackernews stories and comments clearly so the customer can decide which to read to answer their question.
You always keep the maximimum characters per topic within 400 and if there are more than one summaries,
then you create new paragraph with sequence"""
hn_conversation = Conversation()
hn_conversation.add_message("system", hn_system_message)

# Add state and page to the app.
app = rx.App(state=State)
app.add_page(index)
# app.add_page(home)
# app.add_page(signup)
# app.add_page(home)
app.compile()

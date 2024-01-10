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


class State(rx.State):
    """The app state."""

    show_columns = ["Question", "Answer"]
    username: str = ""
    password: str = ""

    prompt: str = ""
    result: str = ""
    status: str = "DE Hackernews Agent"
    metastatus: str = "OpenAI Function Calling Demo"
    is_uploading: bool = False

    async def execute(self):
        self.is_uploading = True
        if self.checked:
            await self.chat_pdf()
        else:
            await self.ask_hn()
        self.is_uploading = False


    async def chat_pdf(self):
        pdf_conversation.add_message("user", self.prompt)
        self.status = "Chunking PDF"
        self.metastatus = f"Splitting PDF in chunks and Summarizing each chunk parallely"
        chat_response, func_call = chat_completion_with_function_execution(
            pdf_conversation.conversation_history, functions=pdf_functions
        )
        print(f"FINAL RESPONSE ****** {chat_response}")
        pdf_conversation.add_message("assistant", chat_response)
        try:
            if chat_response:
                self.status = "PDF Summary Done"
                self.metastatus = f"Function Call: `{func_call}` | Response length: \"{len(chat_response)}\""
            else:
                self.status = "No Response"
                self.metastatus = ""
            self.result = chat_response
        except Exception as e:
            print(e)
            return rx.window_alert("Error occured with OpenAI execution.")

    async def ask_hn(self):
        print(f"PROMPT ****** {self.prompt}")
        hn_conversation.add_message("user", self.prompt)
        self.status = "Processing Query"
        self.metastatus = "Extracting information from hackernews"

        chat_response, func_call = chat_completion_with_function_execution(
            hn_conversation.conversation_history, functions=hn_functions
        )
        hn_conversation.add_message("assistant", chat_response)
        self.is_uploading = False
        try:
            if chat_response:
                self.status = "Response Received"
                self.metastatus = f"Function Call: `{func_call}` | Response length: \"{len(chat_response)}\""
            else:
                self.status = "No Response"
                self.metastatus = ""
            self.result = chat_response
        except Exception as e:
            print(e)
            return rx.window_alert("Error occured with OpenAI execution.")

    checked: bool = False
    is_checked: bool = "HN Conversation!"
    input_placeholder: str = "Ask a question"

    def change_check(self, checked: bool):
        self.checked = checked
        if self.checked:
            self.is_checked = "PDF Summary"
            self.input_placeholder = "Summarize the PDF from the dir `pdfs`"
        else:
            self.is_checked = "HN Conversation!"
            self.input_placeholder = "Ask a question"


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
                    rx.heading(State.is_checked),
                    rx.switch(
                        is_checked=State.checked,
                        on_change=State.change_check,
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder=State.input_placeholder,
                            is_disabled=State.is_uploading,
                            style=style.input_style,
                            on_blur=State.set_prompt,
                        ),
                        rx.button(
                            "Ask", style=style.button_style, on_click=State.execute
                        ),
                        width="900px",
                    ),

                ),
                shadow="lg",
                padding="2em",
                # border_radius="lg",
                width="100%",
            ),
            rx.cond(
                State.is_uploading,
                # rx.progress(is_indeterminate=True, color="blue", width="100%")
                rx.circular_progress(is_indeterminate=True),
                # rx.progress(value=0, width="100%"),
            ),
            rx.center(
                rx.vstack(
                    rx.html(
                        State.result,
                        placeholder="Response",
                        is_disabled=State.is_uploading,
                        width="100%",
                        height="90%",
                        bg="white",
                        color="black",
                        min_height="20em",
                        is_read_only=True,
                        # _focus={"border": 0, "outline": 0, "boxShadow": "none"},
                    ),
                    shadow="lg",
                    padding="1em",
                    border_radius="lg",
                    width="100%",
                    # height="400px",
                ),
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


hn_system_message = """You are a DataEngineering Agent, a helpful assistant to answer user questions.
You summarize the knowledge base and comments clearly in HTML format, 
so the engineers can decide which to read to answer their question.
You always keep the maximum characters per topic within 400 and if there are more than one summaries,
then you create new paragraph with sequence. You always generate the final output as HTML Markdown"""

hn_conversation = Conversation()
hn_conversation.add_message("system", hn_system_message)
pdf_conversation = Conversation()
pdf_conversation.add_message("system", hn_system_message)

# Add state and page to the app.
app = rx.App(state=State)
app.add_page(index)
# app.add_page(home)
# app.add_page(signup)
# app.add_page(home)
app.compile()

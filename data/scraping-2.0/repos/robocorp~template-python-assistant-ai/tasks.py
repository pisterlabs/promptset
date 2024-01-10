from robocorp import vault
from robocorp.tasks import task

from RPA.Assistant.types import WindowLocation, Size
import RPA.Assistant
import openai


assistant = RPA.Assistant.Assistant()
gpt_conversation_display = []
gpt_conversation_internal = []
gpt_model = "gpt-3.5-turbo"


@task
def display_window():
    authorize_openai()

    display_conversation()

    assistant.run_dialog(
        timeout=1800, title="AI Chat", on_top=True, location=WindowLocation.Center
    )


def authorize_openai():
    secrets_container = vault.get_secret("openai")
    openai.api_key = secrets_container["key"]


def show_spinner():
    assistant.clear_dialog()
    assistant.add_loading_spinner(name="spinner", width=60, height=60, stroke_width=8)
    assistant.refresh_dialog()


def ask_gpt(form_data: dict):
    global gpt_conversation_internal

    show_spinner()

    gpt_conversation_internal.append({"role": "user", "content": form_data["input"]})
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=gpt_conversation_internal,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response["choices"][0]["message"]["content"]
    gpt_conversation_internal.append({"role": "assistant", "content": text})
    gpt_conversation_display.append((form_data["input"], text))

    display_conversation()
    assistant.refresh_dialog()


def display_conversation():
    assistant.clear_dialog()
    assistant.add_heading("Conversation")
    for reply in gpt_conversation_display:
        assistant.add_text("You:", size=Size.Small)
        assistant.open_container(background_color="#C091EF", margin=2)
        assistant.add_text(reply[0])
        assistant.close_container()

        assistant.add_text("GPT:", size=Size.Small)
        assistant.open_container(background_color="#A5AACD", margin=2)
        assistant.add_text(reply[1])
        assistant.close_container()

    display_buttons()


def display_buttons():
    assistant.add_text_input("input", placeholder="Send a message", minimum_rows=3)
    assistant.add_next_ui_button("Send", ask_gpt)
    assistant.add_submit_buttons("Close", default="Close")

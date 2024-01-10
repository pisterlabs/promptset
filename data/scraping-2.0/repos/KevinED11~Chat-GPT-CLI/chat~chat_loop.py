"""
This is the main module of this folder
"""
import openai
from rich.prompt import Prompt
from rich.panel import Panel
from gtts import gTTSError


from chat.handle_user_question import handle_user_question
from chat.response_chat import response_chat_gpt
from rich_sources.console import console
from audio.text_to_speech import text_to_speech


def chat_loop(openai_api_key: str, name: str, speech: bool) -> None:
    """ interact with the user
    :param openai_api_key:
    :param name:
    :param speech:
    :return: None
    """
    # API KEY
    openai.api_key = openai_api_key

    # context assistant
    messages: list[dict[str, str]] = [
        {"role": "system",
         "content": "eres un asistente increible y eres el mas inteligente"
         },
    ]

    limit_calls_api: int = 0

    while True:

        # first question for interact with the chat
        user_question: str = Prompt.ask("\n [cyan2 bold]Qué pregunta "
                                        "quieres hacerme [/][bold white]"
                                        ":grey_question:[/]").lower()

        # Use match for compare the question content and make a desicion
        question_results_validate: tuple[str | None, bool] = \
            handle_user_question(user_question=user_question, name=name)

        question, should_exit = question_results_validate

        match should_exit:
            case True:
                break

        match question:
            case None:
                continue

        # Model gpt-3.5-turbo response based on the question
        try:

            text_question_response: str = response_chat_gpt(
                user_question=user_question, messages=messages)

        except (openai.APIError,
                openai.OpenAIError, KeyboardInterrupt,
                UnboundLocalError) as err:

            if not isinstance(err, (KeyboardInterrupt,
                                    UnboundLocalError)):
                print(err)

                break

            limit_calls_api += 1

            if limit_calls_api == 3:
                console.print(
                    "Exceeded maximum number of API calls (3) :cross_mark:",
                    style="bold red")
                break

            print(err)

            continue

        # Print answer to the question
        console.print(Panel.fit(
            text_question_response,
            border_style="yellow1",
            title="Answer to your question",
            width=130,

            style="bold italic white"),
        )

        # Check if speech is true - text to speech
        if speech:
            try:
                text_to_speech(text=text_question_response)

            except (RuntimeError,
                    FileNotFoundError, OSError,
                    KeyboardInterrupt, IOError,
                    gTTSError, TimeoutError) as err:

                if not isinstance(err, KeyboardInterrupt):
                    speech: bool = False

                    console.print(
                        f"Se desactivo la voz debido \
                        a un error del sistema: {err}",
                        style="bold red")

                continue

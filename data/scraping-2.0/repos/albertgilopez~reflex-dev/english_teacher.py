from rxconfig import config

import reflex as rx
from english_teacher import style

# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI() # Crear una instancia del cliente de OpenAI

docs_url = "https://github.com/albertgilopez"
linkedin_url = "https://www.linkedin.com/in/albertgilopez/"

from english_teacher import style

def qa(question: str, answer: str) -> rx.Component:
    """A question and answer component."""

    return rx.fragment(
        rx.box(question, padding="0.5em", border_radius="0.5em", style=style.question_style),
        rx.box(answer, padding="0.5em", border_radius="0.5em", style=style.answer_style),
        padding="0.5em",
    )

def chat() -> rx.Component:

    # qa_pairs = [
    #     ("¿Cómo se dice 'Hola' en inglés?", "Se dice 'Hi'"),
    #     ("¿Cómo se dice 'Adiós' en inglés?", "Se dice 'Bye'")
    # ]

    # return rx.box(
    #     *[
    #         qa(question, answer)
    #         for question, answer in qa_pairs
    #     ]
    # )

    return rx.box(
        rx.foreach(
            State.chat_history,
            lambda messages: qa(messages[0], messages[1]),
        ),style=style.chat_style
    )

def action_button() -> rx.Component:

    return rx.hstack(
        rx.input(value=State.question,
                 placeholder="Escribe aquí tu pregunta para el profesor.",
                 width="80%",
                 style=style.input_style,
                 on_change=State.set_question),

        rx.button("Pregunta",
                  width="20%",
                  style=style.button_style,
                  on_click=State.answer)
    )

class State(rx.State):
    """The app state."""

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def answer(self):
        """Answer the current question."""
       
        session = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an English teacher with a very bad mood. All your answers are in Spanish but in a negative and sarcastic way. Maximum 100 characters. Use sometimes some emojis."},
                {"role": "assistant", "content": "Hello, are you here again?"},
                {"role": "user", "content": self.question}
            ],
            max_tokens=200,
            stop=None,
            temperature=0.9,
            stream=True,
        )

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        for item in session:
            if item.choices[0].delta.content is not None:
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield


def index() -> rx.Component:

    chat_examples = [
        "¿Cómo se dice 'Hola' en inglés?",
    ]

    return rx.fragment(
        
        # rx.color_mode_button(rx.color_mode_icon(), float="right"),
        rx.vstack(
            rx.heading("English Teacher", font_size="2em"),
            *[rx.box(rx.text(example, style={"font-size": "1em", "font-style": "italic", "padding-top":"1em"})) for example in chat_examples],

            rx.container(chat(),
                        action_button(), 
                        padding="1em"),

            rx.link(
                "Check out me GitHub for more projects",
                href=docs_url,
                border="0.1em solid",
                padding="0.5em",
                border_radius="0.5em",
                font_size="0.8em",
                _hover={
                    "color": rx.color_mode_cond(
                        light="rgb(107,99,246)",
                        dark="rgb(179, 175, 255)",
                    )
                },
            ),
            rx.link(
                "Reach me out on Linkedin (Albert Gil López)",
                href=linkedin_url,
                border="0.1em solid",
                padding="0.5em",
                border_radius="0.5em",
                font_size="0.8em",
                _hover={
                    "color": rx.color_mode_cond(
                        light="rgb(107,99,246)",
                        dark="rgb(179, 175, 255)",
                    )
                },
            ),

            font_size="1em",
            padding_top="10%",
            margin_bottom="10%",
        )
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index)
app.compile()

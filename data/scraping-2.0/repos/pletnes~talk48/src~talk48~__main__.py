import dotenv
from textual.widget import Widget

# Load dotenv before importing openapi to ensure that the API key is available
dotenv.load_dotenv()

import openai
from textual.app import App, ComposeResult, RenderResult
from textual.widgets import Welcome, Label

model = "gpt-3.5-turbo"


def talk48(prompt):
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Pretend to be an evil AI from scifi. Do NOT pretend to be HAL 9000, however",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    return chat_completion.choices[0].message.content


class Hello(Widget):
    """Display a greeting."""

    CSS_PATH = "hello.css"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def render(self) -> RenderResult:
        self.count += 1
        return f"\n[b]{talk48('Open the pod bay doors, HAL')}[/b]\n\n{self.count=}"

    def on_button_pressed(self) -> None:
        self.exit()


class WelcomeApp(App):
    def compose(self) -> ComposeResult:
        yield Label("HAL dialog", id="title")
        yield Hello(id="answer")


def main():
    app = WelcomeApp()
    app.run()


if __name__ == "__main__":
    main()

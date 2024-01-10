import openai
import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Input, DataTable, Header, RichLog
from textual.scroll_view import ScrollView
from textual.reactive import reactive
from textual.validation import ValidationResult, Validator

from rich.console import Text

from dotenv import load_dotenv
load_dotenv()


class Termite(App):
    """Main application."""

    CSS_PATH = 'styles.css'
    messages = reactive(list, always_update=True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog()
        yield Input(placeholder="Start typing...")

    # def on_mount(self) -> None:

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.messages = [*self.messages, (event.value)]
        event.input.value = ""

    # def add_message(self, role, content):

    def watch_messages(self, old_messages: list, new_messages: list) -> None:
        print(f"Messages changed from {old_messages} to {new_messages}")
        print(f"New messages: {new_messages[len(old_messages):]}")


if __name__ == "__main__":
    app = Termite()
    app.run()

import openai
import os

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Header, Input, Footer, Static, RichLog
from textual import events
from textual.reactive import reactive

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPTPrompt(Static):
    """A widget to prompt for ChatGPT"""
            
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter your prompt")
        yield Button("Enter", id="enter", variant="primary")

class GPTResponse(Static):
    """A widget to hold ChatGPT responses"""

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True)

    def on_ready(self) -> None:
        text_log = self.query_one(RichLog)
        text_log.write("Waiting for input to query ChatGPT")
        text_log.write("...")

    def on_key(self, event: events.Key) -> None:
        text_log = self.query_one(RichLog)
        # text_log.write(event)

class GPTApp(App):
    """A Textual app to use GPT-4 from the command line"""
    CSS_PATH="interactive.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.Pressed
        if event.button.id == "enter":
            inny = self.query_one(Input)
            prompt = inny.value
            response = openai.ChatCompletion.create(model="gpt-4", messages=[
                {"role": "user", "content": prompt}])
            outty = self.query_one(RichLog)
            outty.write(response["choices"][0]["message"]["content"])

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        yield Footer()
        yield Container(GPTPrompt(), GPTResponse())

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark
    
if __name__ == "__main__":
    app = GPTApp()
    app.run()

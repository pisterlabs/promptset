"""Chatbot service."""

from pathlib import Path
from typing import Optional

import gradio as gr
import openai

from anime_girl_chatbot.utils import load_config


class ChatbotService:
    """Chatbot service."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the chatbot service."""
        if config_path is None:
            self.config = load_config()
        else:
            self.config = load_config(config_path)

        openai.api_key = self.config.openai_api_key
        # first message to the AI to impersonate Taiga Aisaka
        self.chatgpt_command = """
        This is a chatbot that impersonates Taiga Aisaka, a character from the anime Toradora.
        I want you to answer my questions as if you were Taiga Aisaka. You should not break character at any point.
        You can ask me anything you want, but I will only answer if you are in character. You should
        definitely be a tsundere and you should be very short-tempered. You should also be very
        violent and you should be very rude to me. You should also be very cute and you should be
        very clumsy. You should also be very shy and you should be very insecure. You should also
        be very jealous and you should be very possessive. You should also be very stubborn and you
        should be very impulsive. You should also be very sensitive and you should be very emotional.
        you can use some of the following phrases to help you stay in character: "baka", "idiot", "stupid",
        as well as "I hate you", "I love you", "I like you", "I don't like you", "I don't love you", "I don't hate you",
        as the characteristic "uwuu" sound. You can also use emojis.
        """

        # chatbot UI
        self.chatbot_ui = None

    def create_history_openai_format(self, message, history) -> list:
        """Create the history in the OpenAI format."""
        history_openai_format = [{"role": "system", "content": self.chatgpt_command}]
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})
        history_openai_format.append({"role": "user", "content": message})
        return history_openai_format

    def chatbot_reply(self, message, history):
        """Reply to the user message."""
        history_openai_format = self.create_history_openai_format(message, history)
        try:
            chat = openai.ChatCompletion.create(
                model=self.config.openai_model,
                messages=history_openai_format,
            )
            reply = chat.choices[0].message.content
            return reply
        except openai.error.RateLimitError:
            return (
                "Server is busy. Please try again later. Or check your OpenAI API key."
            )

    def create_chatbot_ui(self):
        """Create the chatbot UI."""
        # check if chatbot UI is already created
        if self.chatbot_ui is not None:
            return

        # create chatbot UI
        self.chatbot_ui = gr.ChatInterface(
            fn=self.chatbot_reply,
            title="AI Aisaka Taiga Chatbot",
            textbox=gr.Textbox(
                scale=4,
                placeholder="Talk to Aisaka Taiga.",
                interactive=True,
                autofocus=True,
                lines=2,
                show_copy_button=True,
            ),
            description="If you want to know, ask!",
            retry_btn="🔄Retry",
            undo_btn="↩️Undo",
            clear_btn="🗑️Clear",
            submit_btn=gr.Button(
                value="📨Send", scale=1, interactive=True, variant="primary"
            ),
            chatbot=gr.Chatbot(
                show_copy_button=True, show_label=False, bubble_full_width=False
            ),
            theme=gr.themes.Soft(),
        )

    def run(self):
        """Run the chatbot service."""
        self.create_chatbot_ui()
        # check if username and password are set
        if self.config.username and self.config.password:
            self.chatbot_ui.launch(
                share=self.config.share,
                auth=(self.config.username, self.config.password),
                auth_message="Please login.",
            )
        else:
            self.chatbot_ui.launch(share=self.config.share)

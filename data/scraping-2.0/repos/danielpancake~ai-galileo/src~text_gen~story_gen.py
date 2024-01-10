from dotenv import load_dotenv
from loguru import logger

from utils import toml_interpolate

import os
import re


load_dotenv()

PREFERED_TEXT_GEN_API = os.environ.get("PREFERED_TEXT_GEN_API", "").lower()

# Use Claude API if it's available
if "claude" in PREFERED_TEXT_GEN_API:
    from claude_api import Client

    logger.info("Using Claude API for text generation.")

    class ChatContext:
        """Context manager for Claude API. Creates a new chat and deletes it when the context exits."""

        def __init__(self):
            cookie = os.environ.get("CLAUDE_API_COOKIE")

            if not cookie:
                raise Exception("No COOKIE env variable provided.")

            self.claude_client = Client(cookie)
            self.chat_id = None

        def send_message(self, message: str) -> str:
            return self.claude_client.send_message(
                message,
                self.chat_id,
                timeout=600,
            )

        def __enter__(self):
            self.chat_id = self.claude_client.create_new_chat()["uuid"]
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.claude_client.delete_conversation(self.chat_id)
            self.chat_id = None

else:  # Use OpenAI API if Claude API is not available
    import openai

    logger.info("Using OpenAI API for text generation.")

    class ChatContext:
        """Context manager for ChatGPT API. Stores messages and responeses."""

        def __init__(self):
            self.client = None
            self.messages = []

        def send_message(self, message: str) -> str:
            # Add user message to history
            self.messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            # Send message to ChatGPT with the current conversation history
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
            )
            response_text = resp.choices[0].message.content

            # Add response to history
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

            return response_text

        def __enter__(self):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            self.client = openai.OpenAI()

            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.client = None
            self.messages = []


def generate_episode(config: dict, theme: str, _id: str) -> dict:
    """Generate an episode from a theme, which includes a story, intro, and outro."""

    response = {"theme": theme}

    # Form prompts by interpolating TOML strings
    prompts_items = ["story", "story_intro", "story_outro"]

    prompts = {}
    for item in prompts_items:
        prompts[item] = toml_interpolate(config["prompts"][item], [response, config])

    with ChatContext() as ctx:
        for item in prompts_items:
            response[item] = parse_story(
                ctx.send_message(prompts[item]),
                item,
                _id,
            )

            logger.info(f"Succesfully generated {item} for theme {theme}.")

    return response


def generate_episode_testonly(config: dict, theme: str, _id: str) -> dict:
    """A stub for generate_episode() that doesn't use Claude API. Used for testing."""
    import time

    time.sleep(5)

    response = {"theme": theme}
    for item in ["story", "story_intro", "story_outro"]:
        response[item] = parse_story("Привет, дорогие друзья!", item, _id)

    return response


def parse_story(story_text: str, story_type: str, _id: str) -> list:
    """Parse story text into a list of phrases and actions."""
    phrases = story_text.split("\n\n")  # Extract phrases

    # Remove leading and trailing whitespace
    phrases = [phrase.strip() for phrase in phrases]
    phrases = [p for p in phrases if p]

    voice_lines = 0

    script = []
    for phrase in phrases:
        # Extract and add actions to script
        actions = re.findall(r"\((.*?)\)", phrase)
        for action in actions:
            script.append(
                {
                    "type": "action",
                    "action": action,
                }
            )

        # Remove actions from phrase
        phrase = re.sub(r"\(.*?\)", "", phrase).strip()

        # Add text to script
        if phrase:
            script.append(
                {
                    "type": "text",
                    "text": phrase,
                    "voice": os.path.abspath(
                        f"./output/{_id}/{story_type}/v{voice_lines}.mp3"
                    ),
                }
            )
            voice_lines += 1

    return script

"""Completion module for generating response from OpenAI API."""
from dataclasses import dataclass
from openai import OpenAI, OpenAIError

from app.core.completion.base import ConversionState, Pmessage
from app.core.constants import MAX_TOKENS, MODEL, OPENAI_API_KEY
from app.core.logger.logger import LOGGER


@dataclass
class CompletionData:
    """Dataclass for storing the completion response from OpenAI API."""

    reply_text: str | None
    total_tokens: str | None

    def render(self) -> dict[str, str]:
        """Render the completion response into a dict."""
        return Pmessage("assistant", self.reply_text).render()


def generate_completion_response(state: ConversionState) -> CompletionData:
    """Generate a response from OpenAI API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        prompt = state.conversation.render()
        response = client.chat.completions.create(
            model=MODEL, messages=prompt, temperature=1, top_p=0.9, max_tokens=MAX_TOKENS
        )

        return CompletionData(
            reply_text=response.choices[0].message.content, total_tokens=response.usage.total_tokens
        )

    except OpenAIError as e:
        LOGGER.error(str(e))

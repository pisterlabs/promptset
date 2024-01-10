from openai import OpenAI

import config
from flight_assistant.clients import OpenAISummarizer

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def create_summarizer():
    return OpenAISummarizer(
        openai_client,
        summarize_config=config.OPENAI_SUMMARIZE_SETTINGS,
        prompt=config.OPENAI_SUMMARIZE_PROMPT,
    )

import openai
import click
import time
import tiktoken
from dataclasses import dataclass
from aimojicommit import constants
from aimojicommit.ConfigManager import ConfigManager

@dataclass
class Model:
    id: str
    max_tokens: int

def extract_model_info(model) -> Model:
    return Model(model.id, 16384 if model.id.endswith("16k") else 4096)

class AIManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def get_prompt(self, stats: str, diffs: str, commit_prefix: str):
        return f"""Generate a concise and descriptive git commit message written in present tense and using no emojis apart from the ones provided by the prefix. Interpret the following changes and make a mind-map of the meaning of each change as a list of statements, and combine their concepts in a one-line commit message. Maintain the commit message in one single line and use maximum 72 characters. Your answer must contain text with this format: "##Message: {commit_prefix}: <generated commit message>"
##Git Stats
{stats}
##Git Diffs
{diffs}"""

    def list_available_models(self):
        openai.api_key = self.config_manager.get_openai_api_key()
        models = openai.Model.list().data
        return list(map(extract_model_info, models))

    def get_model_info(self, model_id: str):
        model = Model(model_id, 16384 if model_id.endswith("16k") else 4096)
        return model

    def can_generate_commit_message(
        self, model: Model, stats: str, diffs: str, commit_prefix: str
    ):
        prompt = self.get_prompt(stats, diffs, commit_prefix)
        enc = tiktoken.encoding_for_model(model.id)
        return len(enc.encode(prompt)) <= model.max_tokens - constants.max_response_tokens

    def generate_commit_message(
        self, model: Model, stats: str, diffs: str, commit_prefix: str
    ):
        openai.api_key = self.config_manager.get_openai_api_key()
        prompt = self.get_prompt(stats, diffs, commit_prefix)

        retries = 3  # Number of retries
        for attempt in range(1, retries + 1):
            try:
                response = openai.ChatCompletion.create(
                    model=model.id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=constants.max_response_tokens,
                    temperature=constants.temperature,
                    n=1,
                    stop=None,
                )
                response_text = response.choices[0].message.content.strip()
                response_parts = response_text.split("##Message: ")
                if(len(response_parts) != 2):
                    raise RuntimeError(f"OpenAI response malformed")
                
                commit_message = response_parts[1]
                return commit_message
            except Exception as e:
                click.echo(
                    f"Failed to generate (attempt {attempt}/{retries}): {str(e)}"
                )
                if attempt == retries:
                    raise

                # Retry after a delay
                wait_time = 1  # seconds
                click.echo(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

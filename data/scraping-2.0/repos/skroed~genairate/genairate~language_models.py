import os
import random

from huggingface_hub import InferenceClient
from openai import OpenAI


class LanguageModel(object):
    """Abstract class for language models."""

    def __init__(
        self,
        name: str,
        local: bool = None,
        prompt_type: str = None,
    ):
        self.name = name
        self.local = local

        if prompt_type == "songs":
            self.role = f"""You are an expert music producer with a creative mind, trained in describing songs in appropriate detail and in a consistent way.
            You will be provided a base description of song and you should create a new description of another song in a similar style.
            Make sure that the genre (rock, pop, electronic, ...) of the song is the same as the previous song but the content, the artist names and the song titles
            can be different.
            The description should be 1 sentence long. Also come up with an creative artist name and a song title.
            The output should have the following format:
            <song title>|||<artist name>|||<description>
            Do not use a * or " around the song title, artist name or description.
            """

        elif prompt_type == "moderation":
            fields = [
                "politics",
                "sports",
                "entertainment",
                "gossip",
                "science",
                "tech",
                "economy",
                "health",
                "weather",
                "traffic",
            ]
            selected_field = random.choice(fields)
            self.role = f"""Your role is to be a skilled radio host, trained in talking in an entertaining way.
            Sometimes you can be moody.
            Your task is to moderate between two songs.
            First you should talk one sentence about the previous song.
            Then you should invent 2-3 sentences of entertaining news from the field of {selected_field}.
            Dont tell that these news are fake.
            Then you should find a good transition to the next song.
            The overall text should not be longer than 80 words.
            The name of the previous and next songs will be provided to you in the following format:
            previous: <song title>|||<artist name>|||<description>
            next: <song title>|||<artist name>|||<description>
            """
        else:
            raise ValueError(
                f"Prompt type {prompt_type} not supported. Currently only songs and moderation are supported.",
            )
        self.role = self.role.replace("\n", "")
        self.role = self.role.replace("            ", "")

    def get(self, prompt: str) -> str:
        raise NotImplementedError("LanguageModel is an abstract class.")


class LlamaLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name_or_path: str,
        local: bool = True,
        prompt_type: str = None,
    ):
        """Language model based on the llama model.

        Args:
            model_name_or_path (str): The specific model to use.
            local (bool, optional): If it should run locally. Defaults to True.
            prompt_type (str, optional): The prompt type (see base class). Defaults to None.
        """
        super().__init__(
            name="Llama",
            local=local,
            prompt_type=prompt_type,
        )
        self.inference_client = InferenceClient(model=model_name_or_path)

    def get(self, prompt: str) -> str:
        input_prompt = f"""<s>[INST] <<SYS>>
                    {self.role}
                    <</SYS>>
                    Description: {prompt}
                """
        output_prompt = self.inference_client.text_generation(
            input_prompt,
            max_new_tokens=2000,
        ).split("***")

        return output_prompt


class OpenAiLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        prompt_type: str = None,
        temperature: float = 1.0,
    ):
        """Language model from OpenAI.

        Args:
            model_name (str): The specific model to use.
            local (bool, optional): If it should run locally.
            prompt_type (str, optional): The prompt type (see base class).
        """
        super().__init__(
            name="OpenAi",
            local=False,
            prompt_type=prompt_type,
        )
        self.inference_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.model_name = model_name
        self.temperature = temperature

    def get(self, prompt: str) -> str:
        prompt = prompt.replace("\n", "")
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": prompt},
        ]

        completion = self.inference_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        output_prompt = completion.choices[0].message.content.split("|||")

        return output_prompt


def get_language_model(config: dict) -> LanguageModel:
    """Function to get the language model. Currently only llama and openai are
    supported.

    Args:
        config (dict): The configuration file.

    Returns:
        LanguageModel: The instantiated language model.
    """
    if config["name"] == "Llama":
        language_model = LlamaLanguageModel(
            model_name_or_path=config["model"],
            local=config["local"],
            prompt_type=config["prompt_type"],
        )
    elif config["name"] == "OpenAi":
        language_model = OpenAiLanguageModel(
            model_name=config["model"],
            prompt_type=config["prompt_type"],
            temperature=config["temperature"],
        )
    else:
        raise ValueError(f"Language model {config['name']} not supported.")
    return language_model

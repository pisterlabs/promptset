import os
from typing import TypeVar, Dict, List

import openai
from dotenv import load_dotenv

from llms.llm_models import OpenAIModel

from src.llms.token_calculator import TokenCalculator

AIObject = TypeVar("AIObject")

load_dotenv()
OPEN_AI_ORG = os.environ["OPEN_AI_ORG"]
OPEN_AI_KEY = os.environ["OPEN_AI_KEY"]

assert OPEN_AI_ORG and OPEN_AI_KEY, f"Must supply value for {f'{OPEN_AI_ORG=}'.split('=')[0]} " \
                                    f"and {f'{OPEN_AI_KEY=}'.split('=')[0]} in .env"
openai.organization = OPEN_AI_ORG
openai.api_key = OPEN_AI_KEY


class LLMManager:
    """
    Interface for all AI utility classes.
    """

    @staticmethod
    def make_completion(prompt: str, temperature: float = 0, model: OpenAIModel = OpenAIModel.GPT4,
                        conversation_history: List[Dict] = None) -> List[Dict]:
        """
        Makes a request to completion a model
        :param prompt: The prompt to make completion for.
        :param temperature: The temperature to run the model at.
        :param model: The OpenAI model to use.
        :param conversation_history: Contains all the previous responses and messages between AI and Human
        :return: The response from open AI.
        """

        assert isinstance(model, OpenAIModel), f"Expected OpenAIModel to be passed in but got {model}."

        conversation_history = [] if not conversation_history else conversation_history
        conversation_history.append({"role": "user", "content": prompt})

        if model == OpenAIModel.GPT4:
            all_prompts = [p["content"] for p in conversation_history]
            max_tokens = TokenCalculator.calculate_max_tokens(OpenAIModel.GPT4, "".join(all_prompts))
        else:
            max_tokens = model.get_max_tokens()

        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": model.value,
            "messages": conversation_history}
        res = openai.ChatCompletion.create(**params)
        res_text = res.choices[0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": res_text})
        return conversation_history

from abc import ABC, abstractmethod
from typing import Literal

import openai
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from vertexai.preview.language_models import TextGenerationModel


def _load_credentials() -> dict:
    """Load credentials dict from `conf/local/credentials.yml` file.

    Returns:
        dict: credentials
    """
    conf_path = str(settings.CONF_SOURCE)
    conf_loader = ConfigLoader(conf_source=conf_path)
    credentials = conf_loader["credentials"]

    return credentials


def _get_openai_api_key(credentials: dict) -> str:
    """Retrieve OpenAI API key.

    Returns:
        str: Azure OpenAI API key
    """
    openai_api_key = credentials["openai_api_key"]

    return openai_api_key


def _get_azure_openai_api_credentials(credentials: dict) -> tuple[str]:
    """Retrieve Azure OpenAI API credentials.

    Returns:
        tuple[str]: Azure OpenAI API credentials
    """
    azure_openai_api_key = credentials["azure_openai_api_key"]
    azure_openai_api_endpoint = credentials["azure_openai_api_endpoint"]

    return azure_openai_api_key, azure_openai_api_endpoint


class APIRequest(ABC):
    def __init__(
        self,
        mode: Literal["explain", "summarize"],
        input_text: str,
        instructions: dict[str],
        max_tokens: dict[int],
        model: str,
    ):
        """Initialize APIRequest class.

        Args:
            mode (Literal[&quot;explain&quot;, &quot;summarize&quot;]): execution mode; either `explain` or `summarize`
            input_text (str): text to be explained or summarized
            instructions (dict[str]): additional instructions for the model depending on the `mode`
            max_tokens (dict[int]): maximum number of tokens to generate depending on the `mode`
            model (str, optional): OpenAI model name.
        """
        self.mode = mode
        self.input_text = input_text
        self.instructions = instructions
        self.max_tokens = max_tokens
        self.model = model
        self.response = None

    @abstractmethod
    def execute_prompt(self) -> None:
        """Execute prompt and store results as an API-specific object."""
        pass

    @abstractmethod
    def extract_answer(self) -> str:
        """Extract text answer from the API-specific response object.

        Returns:
            str: text answer from model
        """
        pass


class OpenAIAPIRequest(APIRequest):
    def execute_prompt(self) -> None:
        openai.api_type = "open_ai"
        openai.api_version = None
        openai.api_key = _get_openai_api_key(_load_credentials())
        openai.api_base = "https://api.openai.com/v1"

        self.response = openai.Completion.create(
            model=self.model,
            prompt=self.instructions[self.mode] + self.input_text,
            temperature=0,
            max_tokens=self.max_tokens[self.mode],
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def extract_answer(self) -> str:
        answer = self.response.get("choices")[0].get("text").strip()

        return answer


class VertexAIPaLMAPIRequest(APIRequest):
    def execute_prompt(self) -> None:
        llm_model = TextGenerationModel.from_pretrained(self.model)

        self.response = llm_model.predict(
            prompt=self.instructions[self.mode] + self.input_text,
            max_output_tokens=self.max_tokens[self.mode],
            temperature=0.0,
            top_p=1.0,
            top_k=40,
        )

    def extract_answer(self) -> str:
        answer = self.response.text

        return answer


class AzureOpenAIAPIRequest(APIRequest):
    def execute_prompt(self) -> None:
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key, openai.api_base = _get_azure_openai_api_credentials(
            _load_credentials()
        )

        self.response = openai.Completion.create(
            engine=self.model,
            prompt=self.instructions[self.mode] + self.input_text,
            temperature=0,
            max_tokens=self.max_tokens[self.mode],
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def extract_answer(self) -> str:
        answer = self.response.get("choices")[0].get("text").strip()

        return answer


class PaLMAPIRequest(APIRequest):
    # To be implemented when granted access from waitlist
    pass

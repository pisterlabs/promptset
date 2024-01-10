from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.llms.ollama import Ollama
from langchain.chat_models.openai import ChatOpenAI
from typing import Dict, List, Union, Any
import subprocess
import json
import re
import os


class InOut(ABC):
    """InOut class for interacting with LLMs."""

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

    def __init__(self, llm: Union[BaseLLM, BaseChatModel]) -> None:
        """Initializes InOut class."""
        self._llm = llm

    def generate_random_article_idea(self, category_injection: str) -> str:
        """
        Generates a random article idea.

        Args:
            phrase_init (str): Word to initialize the article idea with.
            category_injection (str): Category to inject into prompt for latent space activation.

        Returns:
            str: Random article idea.
        """
        messages = [
            SystemMessage(content=self._load_prompt("generate_random_article_idea")),
            HumanMessage(
                content=f"Request: 'Please give me one idea exactly', Broad Category Idea: '{category_injection}'"
            ),
        ]
        while True:
            try:
                generated_text = self._parse_single_line_output(
                    self._invoke_model(input=messages)
                )
                assert (
                    len(generated_text) > 0
                ), "Output is empty. Please try again with a different category."
                return generated_text
            except Exception as e:
                print(str(e))

    def write_news_article(
        self, article_idea: str, category_constraint: List[str]
    ) -> Dict:
        """
        Writes a news article.

        Args:
            article_idea (str): Idea for article.
            category_constraint (List[str]): List of categories to constrain output selection to.

        Returns:
            Dict: News article.

        Return Format:
            {
                "title": "<Long, Specific, and SEO Optimized Title>",
                "category_list": <["category_1", "category_2"]>,
                "header_img_description": "<hyper-detailed description of image to use for header image>",
                "body": "<## Body in Markdown\\n\\nShould use expressive markdown syntax with proper \\"escape\\" characters>"
            }
        """
        messages = [
            SystemMessage(content=self._load_prompt("write_news_article")),
            HumanMessage(
                content=f"Article Idea: '{article_idea}', Categories to Choose From: {category_constraint}"
            ),
        ]
        while True:
            try:
                generated_text = self._invoke_model(input=messages)
                loaded_json = self._parse_json_output(generated_text)
                self._raise_for_bad_json_output(
                    input_json=loaded_json,
                    expected_keys_to_type={
                        "title": str,
                        "category_list": list,
                        "header_img_description": str,
                        "body": str,
                    },
                )
                assert len(loaded_json["title"]) > 0, "Error: Title is empty."
                assert len(loaded_json["title"]) < 150, "Error: Title is too long."
                assert len(loaded_json["category_list"]) in list(
                    range(1, 4)
                ), "Error: Wrong number of categories."
                assert all(
                    category in category_constraint
                    for category in loaded_json["category_list"]
                ), "Error: Category list must be a subset of the category constraint."
                assert (
                    len(loaded_json["header_img_description"]) > 0
                ), "Error: Header image description is empty."
                assert len(loaded_json["body"]) > 0, "Error: Body is empty."
                return loaded_json
            except Exception as e:
                print(str(e))

    def _invoke_model(self, input: Any) -> str:
        """Abstracts LLM / ChatModel Call"""
        if isinstance(self._llm, BaseLLM):
            return self._llm.invoke(input=input)
        if isinstance(self._llm, BaseChatModel):
            return self._llm.invoke(input=input).content

    def _load_prompt(self, prompt_name: str) -> str:
        """Loads prompt from file."""
        with open(f"{self.PROMPTS_DIR}/{prompt_name}.txt", "r") as f:
            return f.read()

    @staticmethod
    def _parse_single_line_output(input: str) -> str:
        """
        Parses single line output from LLM.

        Args:
            input (str): Input string to clean.

        Returns:
            str: Cleaned string.
        """
        input = input.replace("<|im_end|>", "")
        input = input.replace("\n", " ")
        input = input.replace('"', "")
        input = input.replace(".", "")
        input = input.strip()
        return input

    @staticmethod
    def _parse_json_output(json_string: str) -> Dict:
        """
        Parses JSON output from LLM.

        Args:
            json_string (str): JSON string to clean.

        Returns:
            Dict: Cleaned JSON.

        Raises:
            JSONDecodeError: If JSON string is not valid.
        """
        json_string = json_string.replace("<|im_end|>", "")
        removed_internal_newlines = re.sub(
            r'\"([^"]*)\"',  # match anything in quotes including quotes
            lambda match: match.group(0).replace("\n", "\\n"),
            json_string,
        )
        return json.loads(removed_internal_newlines)

    @staticmethod
    def _raise_for_bad_json_output(
        input_json: Dict, expected_keys_to_type: Dict
    ) -> None:
        """
        Raises for bad JSON format.

        Args:
            input_json (Dict): Loaded JSON.
            expected_keys_to_type (Dict): Expected keys to type mapping.

        Raises:
            AssertionError: If JSON is not valid.
        """
        assert isinstance(input_json, dict), "Output is not valid JSON."
        assert set(input_json.keys()) == set(
            expected_keys_to_type.keys()
        ), "Output keys do not match expected keys."
        for input_key, input_value in input_json.items():
            assert (
                type(input_value) == expected_keys_to_type[input_key]
            ), f"Value for key '{input_key}' is not of expected type '{expected_keys_to_type[input_key]}'."

    @staticmethod
    @abstractmethod
    def kill() -> None:
        """Kills LLM runner process."""
        pass


class OllamaInOut(InOut):
    def __init__(self, model_name: str, temperature: int):
        print(f"Pulling Ollama model: {model_name}")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Running Ollama model: {model_name}")
        llm = Ollama(model=model_name, temperature=temperature)
        super().__init__(llm)

    @staticmethod
    def kill():
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ollama-runner"], capture_output=True, text=True
            )
            pid = result.stdout.strip()
            if pid:
                subprocess.run(["kill", "-9", pid], check=True)
                print(f"Ollama runner with PID {pid} has been successfully killed.")
            else:
                print("Ollama runner process not found.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


class OpenAIInOut(InOut):
    def __init__(self, model_name: str, temperature: int, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        super().__init__(llm)

    @staticmethod
    def kill():
        pass  # not local process

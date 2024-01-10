import requests

from pygitai.common.config import config
from pygitai.common.llm.base import LLMBase, ParserBase, PromptLine
from pygitai.common.logger import get_logger

logger = get_logger(__name__, config.logger.level)


class HuggingFaceParser(ParserBase[requests.Response, str, str]):
    @staticmethod
    def parse_response(response, prompt):
        """Parse the response from OpenAI"""
        return " ".join([data["generated_text"] for data in response.json()])

    @staticmethod
    def parse_prompt(input_data: tuple[PromptLine, ...]):
        """Parse the input data and return a list of dict"""
        return "\n\n".join([row.text for row in input_data])


class HuggingFace(LLMBase[str, str]):
    config = config.hugging_face
    llm_parser = HuggingFaceParser

    @classmethod
    def exec_prompt(cls, prompt, model):
        payload = {
            "inputs": prompt,
        }
        logger.info("Wait for hugging-face response")
        logger.debug(f"Send Payload to hugging-face: {payload}")
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {cls.config.api_token}"},
            json=payload,
        )
        response.raise_for_status()
        logger.info("HuggingFace response received")
        logger.debug(f"HuggingFace response: {response.json()}")

        parsed_llm_response = cls.llm_parser.parse_response(
            prompt=prompt,
            response=response,
        )
        full_context = f"{prompt}\n\n{parsed_llm_response}"
        return parsed_llm_response, full_context

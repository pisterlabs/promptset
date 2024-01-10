import json
from typing import Dict
from langchain.llms.sagemaker_endpoint import LLMContentHandler


class SageMakerContentHandler(LLMContentHandler):
    """
    SageMaker handler for a T5-XXL endpoint
    Args:
        LLMContentHandler(langchain.llms.sagemaker_endpoint): default content
            handler for LLMs
    """

    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


class AI21SageMakerContentHandler(LLMContentHandler):
    """
    SageMaker handler for a J2 Jumbo Instruct endpoint
    Args:
        LLMContentHandler(langchain.llms.sagemaker_endpoint): default content
            handler for LLMs
    """

    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["completions"][0]["data"]["text"]

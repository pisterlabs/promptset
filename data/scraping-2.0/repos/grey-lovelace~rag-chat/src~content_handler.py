from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {
                "inputs": [
                    [
                        {"role": "system", "content": "You are a kind robot."},
                        {"role": "user", "content": prompt},
                    ]
                ],
                "parameters": {**model_kwargs},
            }
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

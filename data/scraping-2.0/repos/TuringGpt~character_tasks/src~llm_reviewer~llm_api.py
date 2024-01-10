import json
from openai import OpenAI
from src.llm_reviewer.utils import load_config


class LLMAPIFactory:
    def __init__(self, secrets_file_path: str):
        self.secrets_file_path = secrets_file_path
        self._api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        config = load_config(self.secrets_file_path)
        return config["openai_api_key"]

    def get(self) -> OpenAI:
        return OpenAI(api_key=self._api_key)


def make_llm_request(
    client,
    messages: list[dict[str, str]],
    model: str  = None,
    temperature: float = 1.0,
    max_tokens: int = 4000,
    response_format: str  = None,
    retries: int = 3,
) -> str:
    if response_format not in [{"type": "json_object"}, None]:
        raise ValueError(
            "Unsupported response format. Only 'json_object' or None is allowed."
        )
    if response_format is not None and model not in [
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-1106",
    ]:
        raise ValueError("Model not supported for response_format argument.")

    for retry in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            if completion.choices[0].finish_reason != "stop":
                raise Exception(
                    "The conversation was stopped for an unexpected reason."
                )

            if response_format == {"type": "json_object"}:
                try:
                    return json.loads(completion.choices[0].message.content)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON response. Full completion:")
                    print(completion)
                    raise e
            else:
                return completion.choices[0].message.content
        except Exception as e:
            print(
                f"Attempt {retries - retry} of {retries} failed with error: {e}. Retrying..."
            )
    raise Exception("All attempts failed.")

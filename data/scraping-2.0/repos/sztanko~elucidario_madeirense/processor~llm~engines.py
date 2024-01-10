import logging
from abc import ABC, abstractmethod
from openai import OpenAI
import os
import json

import boto3
from botocore.config import Config

from processor.utils.extract_json import extract_json_simple

CHARS_PER_TOKEN = 3.5

DEFAULT_SIMPLE_GPT_MODEL = "gpt-3.5-turbo-1106"
DEFAULT_COMPLEX_GPT_MODEL = "gpt-4-1106-preview"  # "gpt-3.5-turbo-1106"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)




class AIEngine(ABC):
    def __init__(self, instructions: str):
        self.instructions = instructions

    @abstractmethod
    def get_max_length(self) -> int:
        return 0

    @abstractmethod
    def _submit_message(self, message: str) -> str:
        return ""

    def submit_message(self, message: str) -> str:
        if len(message) > self.get_max_length():
            logging.warn(f"Message length {len(message)} exceeds maximum length {self.get_max_length()}")
        logging.info(f"Submitting message of length {len(message)}")
        # logging.info(message)
        result = self._submit_message(message)
        out = extract_json_simple(result)
        # logging.info("Distilled fixed json output is:")
        # logging.info(out)
        return out


class OpenAIEngine(AIEngine):
    def __init__(
        self,
        instructions: str,
        simple_gpt_model=DEFAULT_SIMPLE_GPT_MODEL,
        complex_gpt_model=DEFAULT_COMPLEX_GPT_MODEL,
    ):
        super().__init__(instructions)
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.simple_gpt_model = simple_gpt_model
        self.complex_gpt_model = complex_gpt_model
        logging.info("OpenAI client successfully created!")
        # logging.info("Instructions are")
        # logging.info(self.instructions)

    def get_max_length(self) -> int:
        return 4096 * CHARS_PER_TOKEN

    def determine_model(self, message):
        if "......." in message:
            return self.complex_gpt_model
        else:
            return self.simple_gpt_model

    def _submit_message(self, message: str) -> str:
        model = self.determine_model(message)
        logging.info(f"Using model: {model}")
        messages = [{"role": "system", "content": self.instructions}, {"role": "user", "content": message}]
        # logging.info("Submitting a message to OpenAI API")
        # logging.info(messages)
        completion = self.client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, messages=messages, temperature=0.1
        )
        body_text = completion.choices[0].message.content
        return body_text


class GPT3Engine(OpenAIEngine):
    def __init__(
        self,
        instructions: str,
    ):
        super().__init__(instructions)

    def determine_model(self, message):
        return "gpt-3.5-turbo-1106"


class GPT4Engine(OpenAIEngine):
    def __init__(
        self,
        instructions: str,
    ):
        super().__init__(instructions)

    def determine_model(self, message):
        return "gpt-4-1106-preview"


AWS_REGION = "us-east-1"
CLAUDE_MAX_TOKENS = 4096
CLAUDE_MODEL = "anthropic.claude-v2:1"


class ClaudeEngine(AIEngine):
    def __init__(self, instructions: str):
        super().__init__(instructions)
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        session_kwargs = {
            "region_name": AWS_REGION,
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        }
        retry_config = Config(
            region_name=AWS_REGION,
            read_timeout=900,
            connect_timeout=900,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        self.session = boto3.Session(**session_kwargs)
        self.client = self.session.client(service_name="bedrock-runtime", config=retry_config, **session_kwargs)
        logging.info("boto3 Bedrock client successfully created!")
        logging.info(self.client._endpoint)
        # logging.info(dir(self.client))
        # logging.info(self.client.list_foundation_models())

    def get_max_length(self) -> int:
        return CLAUDE_MAX_TOKENS * CHARS_PER_TOKEN

    def determine_model(self, message):
        return CLAUDE_MODEL

    def _submit_message(self, message: str) -> str:
        model = self.determine_model(message)
        prompt_data = f"Human: {self.instructions}\n\nHere is the article html:\n{message}\n\nAssistant:"
        body = json.dumps(
            {
                "prompt": prompt_data,
                "max_tokens_to_sample": CLAUDE_MAX_TOKENS,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 80,
            }
        )
        # model = "anthropic.claude-instant-v1"
        logging.info(f"Using model: {model}")
        # logging.info(f"{prompt_data}")

        response = self.client.invoke_model_with_response_stream(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
            # anthropic_version="bedrock-2023-05-31",
        )
        stream = response.get("body")
        out = []
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    text = json.loads(chunk.get("bytes").decode())["completion"]
                    out.append(text)
                    # print(text, end="")
        # return json.loads(response.get("body").read())

        response = "".join(out)

        return response

ENGINE_MAP = {
    "gpt3": GPT3Engine,
    'gpt4': GPT4Engine,
    'claude': ClaudeEngine,
}

def get_engine(engine_str: str) -> AIEngine:
    return ENGINE_MAP.get(engine_str)


if __name__ == "__main__":
    ai = ClaudeEngine("Hello!")
    print(ai.submit_message("Meselj nekem egy viccet!"))

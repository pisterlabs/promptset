import abc
import json
import os
import random
from typing import Dict, List

from jinja2 import Template

from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import (
    OpenAiAuthenticationError,
    BedRockAuthenticationError,
)


class LLM:
    def __init__(self, model_name):
        from pydantic_prompter.settings import Settings

        self.settings = Settings()
        self.model_name = model_name

    def debug_prompt(self, messages: List[Message], scheme: dict):
        raise NotImplementedError

    def call(self, messages: List[Message], scheme: Dict) -> str:
        raise NotImplementedError

    @classmethod
    def get_llm(cls, llm: str, model_name: str):
        if llm == "openai":
            llm_inst = OpenAI(model_name)
        elif llm == "bedrock" and model_name.startswith("anthropic"):
            logger.debug("Using bedrock provider with Anthropic model")
            llm_inst = BedRockAnthropic(model_name)
        elif llm == "bedrock" and model_name.startswith("cohere"):
            logger.debug("Using bedrock provider with Cohere model")
            llm_inst = BedRockCohere(model_name)
        elif llm == "bedrock" and model_name.startswith("meta"):
            logger.debug("Using bedrock provider with Cohere model")
            llm_inst = BedRockLlama2(model_name)
        else:
            raise Exception(f"Model not implemented {llm}, {model_name}")
        logger.debug(
            f"Using {llm_inst.__class__.__name__} provider with model {model_name}"
        )
        return llm_inst


class OpenAI(LLM):
    @staticmethod
    def to_openai_format(msgs: List[Message]):
        openai_msgs = [item.model_dump() for item in msgs]
        return openai_msgs

    def debug_prompt(self, messages: List[Message], scheme: dict) -> str:
        return json.dumps(self.to_openai_format(messages), indent=4, sort_keys=True)

    def call(self, messages: List[Message], scheme: dict) -> str:
        from openai import OpenAI
        from openai import AuthenticationError, APIConnectionError

        _function_call = {
            "name": scheme["name"],
        }
        logger.debug(f"Openai Functions: \n [{scheme}]")
        logger.debug(f"Openai function_call: \n {_function_call}")
        messages_oai = self.to_openai_format(messages)
        try:
            client = OpenAI(api_key=self.settings.openai_api_key)
            chat_completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages_oai,
                functions=[scheme],
                function_call=_function_call,
                temperature=random.uniform(0.3, 1.3),
            )
        except (AuthenticationError, APIConnectionError) as e:
            raise OpenAiAuthenticationError(e)
        return chat_completion.choices[0].message.function_call.arguments


class BedRock(LLM, abc.ABC):
    @abc.abstractmethod
    def _strip_wrapping_garbage(self, body: str) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _template_path(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _stop_sequence(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def format_messages(self, msgs: List[Message]) -> str:
        raise NotImplementedError

    def build_prompt(self, messages: List[Message], scheme: dict):
        if "prompt_templates" not in self._template_path:
            logger.info(f"Using custom prompt from {self._template_path}")
        ant_template = open(self._template_path).read()
        ant_scheme = json.dumps(scheme["parameters"], indent=4)
        ant_msgs = self.format_messages(messages)
        template = Template(ant_template, keep_trailing_newline=True)
        content = template.render(schema=ant_scheme, question=ant_msgs).strip()
        return content

    def debug_prompt(self, messages: List[Message], scheme: dict) -> str:
        return self.build_prompt(messages, scheme)

    def call(self, messages: List[Message], scheme: dict) -> str:
        content = self.build_prompt(messages, scheme)

        body = json.dumps(
            {
                "max_tokens_to_sample": 8000,
                "prompt": content,
                "stop_sequences": [self._stop_sequence],
                "temperature": random.uniform(0, 1),
            }
        )
        logger.debug(f"Request body: \n{body}")
        try:
            import boto3

            os.environ["AWS_ACCESS_KEY_ID"] = self.settings.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.aws_secret_access_key

            session = boto3.Session(
                profile_name=self.settings.aws_profile,
                region_name=self.settings.aws_default_region,
            )
            client = session.client("bedrock-runtime")
            response = client.invoke_model(
                body=body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            raise BedRockAuthenticationError(e)

        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        res = self._strip_wrapping_garbage(response_body.get("completion"))
        return res


class BedRockAnthropic(BedRock):
    def _strip_wrapping_garbage(self, body: str):
        body = body.replace("</json>", "")
        left = body.find("{")
        right = body.rfind("}")
        j = body[left : right + 1]
        return j

    @property
    def _template_path(self):
        return self.settings.template_paths.anthropic

    @property
    def _stop_sequence(self):
        return "Human:"

    def format_messages(self, msgs: List[Message]) -> str:
        role_converter = {"user": "Human", "system": "Human", "assistant": "Assistant"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)


class BedRockCohere(BedRock):
    def _strip_wrapping_garbage(self, body: str) -> str:
        body = body.replace("</json>", "")
        left = body.find("{")
        right = body.rfind("}")
        j = body[left : right + 1]
        return j

    @property
    def _template_path(self) -> str:
        return self.settings.template_paths.cohere

    @property
    def _stop_sequence(self) -> str:
        return "User:"

    def format_messages(self, msgs: List[Message]) -> str:
        role_converter = {"user": "User", "system": "System", "assistant": "Chatbot"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)

    def call(self, messages: List[Message], scheme: dict) -> str:
        content = self.build_prompt(messages, scheme)

        body = json.dumps(
            {
                "prompt": content,
                "stop_sequences": [self._stop_sequence],
                "temperature": random.uniform(0, 1),
            }
        )
        logger.debug(f"Request body: \n{body}")
        try:
            import boto3

            os.environ["AWS_ACCESS_KEY_ID"] = self.settings.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.aws_secret_access_key

            session = boto3.Session(
                profile_name=self.settings.aws_profile,
                region_name=self.settings.aws_default_region,
            )
            client = session.client("bedrock-runtime")
            response = client.invoke_model(
                body=body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            raise BedRockAuthenticationError(e)

        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        res = self._strip_wrapping_garbage(response_body["generations"][0]["text"])
        return res


class BedRockLlama2(BedRock):
    def _strip_wrapping_garbage(self, body: str) -> str:
        body = body.replace("</json>", "")
        left = body.find("{")
        right = body.rfind("}")
        j = body[left : right + 1]
        return j

    @property
    def _template_path(self) -> str:
        return self.settings.template_paths.llama2

    @property
    def _stop_sequence(self) -> str:
        return "</s>"

    def format_messages(self, msgs: List[Message]) -> str:
        output = []
        for msg in msgs:
            if msg.role == "system":
                output.append(f"<<SYS>> {msg.content} <</SYS>>")
            if msg.role == "assistant":
                output.append(f"{msg.content} </s>")
            if msg.role == "user":
                output.append(f"{msg.content}  [/INST]")
        return "\n".join(output)

    def call(self, messages: List[Message], scheme: dict) -> str:
        content = self.build_prompt(messages, scheme)

        body = json.dumps(
            {
                "max_gen_len": 2048,
                "prompt": content,
                "temperature": random.uniform(0, 1),
            }
        )
        logger.debug(f"Request body: \n{body}")
        try:
            import boto3

            os.environ["AWS_ACCESS_KEY_ID"] = self.settings.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.aws_secret_access_key

            session = boto3.Session(
                profile_name=self.settings.aws_profile,
                region_name=self.settings.aws_default_region,
            )
            client = session.client("bedrock-runtime")
            response = client.invoke_model(
                body=body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            logger.warning(e)
            raise BedRockAuthenticationError(e)

        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        res = self._strip_wrapping_garbage(response_body.get("generation"))
        return res

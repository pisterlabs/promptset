import json
import os
from pathlib import Path
import random
import string
import time
from typing import Any, Dict, List

from flask import Request, jsonify, request
from langchain.schema import BaseMessage

from openserver.core.utils import extract_json_from_string, base_messages_to_default, logger
from openserver.core.config import ChatConfig, PromptConfig
from openserver.core.llm_models.base import LLmInputInterface
from openserver.core.llm_models.llm_model_factory import LLMFactory
from openserver.core.utils.cost import completion_price_calculator
from openserver.server.app import app
from openserver.server.utils import llm_result_to_str, num_tokens_from_string


class ChatCompletionsRequest:
    def __init__(self, request: Request):
        try:
            self.model: str = request.get_json().get("model", "gpt-3.5-turbo")
            self.stream: bool = request.get_json().get("stream", False)
            self.api_key: str | None = request.get_json().get("api_key") or (request.authorization.token if request.authorization is not None else None)
            self.messages: List[Dict[str, Any]
                                ] = request.get_json().get("messages")
            self.functions = request.get_json().get("functions")
            self.n_gpu_layers: int = request.get_json().get("n_gpu_layers", 99)
            self.temperature: float = request.get_json().get("temperature", 0.4)
            self.max_tokens: int = request.get_json().get("max_tokens", 1000)
            self.top_p: int = request.get_json().get("top_p", 1)
            self.cache: bool = request.get_json().get("cache", False)
            self.n_ctx: int = request.get_json().get("n_ctx", 8196)
        except Exception as e:
            return jsonify({'reason': "request data error", 'error': str(e)}), 500


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    try:
        
        request_data = ChatCompletionsRequest(request)

        available_functions = False

        if "functions" in request.get_json():
            available_functions = True

        configs = ChatConfig(with_envs=True)
        provider = configs.get_chat_providers(
            request_data.model, available_functions)

        logger.info(provider)

        chat_input = LLmInputInterface(
            api_key=request_data.api_key or provider.args.get("api_key_name"),
            model=provider.key or provider.name,
            model_kwargs={
                "chat_format": "mistral",
            },
            streaming=request_data.stream,
            n_gpu_layers=request_data.n_gpu_layers,
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens,
            top_p=request_data.top_p,
            cache=request_data.cache,
            n_ctx=request_data.n_ctx,
            base_url=provider.args.get("base_url")
        )

        messages = [BaseMessage(
            type=message["role"], content=message["content"]) for message in request_data.messages]
        messages = base_messages_to_default(messages)

        if available_functions is True:
            configs = PromptConfig()
            new_messages = configs.extract_text(configs.prompt_template(
            ), prompt=messages[-1].content, functions=request_data.functions)
            messages.pop()
            messages = messages + new_messages

            ROOT_DIR: str = os.path.dirname(Path(__file__).parent.parent.parent)
            chat_input.grammer_path = ROOT_DIR + "/docs/json.gbnf"
            chat_input.f16_kv = True

        chatProvider = LLMFactory.get_chat_model(
            input=chat_input, provider_name=provider.provider)
        response = chatProvider.compelete(
            prompts=[messages])
        response_str = llm_result_to_str(response)

        completion_id = "".join(random.choices(
            string.ascii_letters + string.digits, k=28))
        completion_timestamp = int(time.time())

        if not request_data.stream:
            inp_token = num_tokens_from_string(
                "".join([message.content for message in messages]))
            out_token = num_tokens_from_string(response_str)
            function_out = None

            if available_functions is True:
                function_out = extract_json_from_string(response_str)

            res = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion",
                "created": completion_timestamp,
                "model": provider.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_str,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inp_token,
                    "completion_tokens": out_token,
                    "total_tokens": inp_token + out_token,
                    "cost": "{:.6f}".format(completion_price_calculator(provider.cost.input, provider.cost.output, inp_token, out_token))
                },
            }
            if function_out is not None and function_out != "" and isinstance(function_out, dict):
                res["choices"][0]["message"]["content"] = None
                res["choices"][0]["message"]["function_call"] = add_to_arguments(
                    function_out)
                res["choices"][0]["message"]["content"] = f'{res["choices"][0]["message"]["function_call"]}'
                res["choices"][0]["finish_reason"] = "function_call"
            return res

        def streaming():
            for chunk in response:
                completion_data = {
                    "id": f"chatcmpl-{completion_id}",
                    "object": "chat.completion.chunk",
                    "created": completion_timestamp,
                    "model": provider.name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk,
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                content = json.dumps(completion_data, separators=(",", ":"))
                yield f"data: {content}"

                time.sleep(0.1)

            end_completion_data = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion.chunk",
                "created": completion_timestamp,
                "model": provider.name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            content = json.dumps(end_completion_data, separators=(",", ":"))
            yield f"data: {content}"

        return app.response_class(streaming(), mimetype="text/event-stream") # type: ignore
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/chat/completions", methods=["GET"])
def get_chat_models():
    try:
        configs = ChatConfig()
        for provider in configs.chat_providers.providers:
            provider.api_key = ""
            provider.args = dict(filter(lambda item: item[0] not in [
                                           'api_key', 'api_key_name'], provider.args.items()))
        return jsonify(configs.chat_providers.model_dump())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def add_to_arguments(data: dict):
    if 'parameters' in data:
        data['arguments'] = data.pop('parameters')

    if 'properties' in data:
        data['arguments'] = data.pop('properties')

    if 'arguments' not in data:
        data['arguments'] = {}
    
    if isinstance(data["arguments"], str):
        data["arguments"] = extract_json_from_string(data["arguments"]) or {}

    for key, value in data.items():
        if key != 'arguments' and key != 'name':
            data['arguments'][key] = value

    data["arguments"] = json.dumps(data["arguments"])

    return data

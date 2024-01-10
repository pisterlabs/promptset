
from typing import Any, Dict, Mapping
import openai
from schema.messages import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage
from schema.output import ChatGeneration, ChatResult


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)

class ChatModel(object):
    def __init__(self, model_name='gpt-3.5-turbo', **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.0)
        self.function_call = kwargs.get('function_call', 'auto')
        self.max_tokens = kwargs.get('max_tokens', None)
        self.top_p = kwargs.get('top_p', 1.0)
        self.n = kwargs.get('n', 1)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {
            "token_usage": response["usage"], 'model_name': self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def __call__(self, messages, functions, stop=None):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            functions=functions,
            function_call=self.function_call,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=stop
        )
        return self._create_chat_result(response)

from enum import Enum
from typing import Dict, Optional, Tuple, Iterator

import tiktoken
from PySide6.QtCore import QTranslator

from backend.agents.base_agent import BaseAgent, BaseTrigger, BaseResult
from backend.models import Error
from backend.tools.utils import logger
from setting.setting_reader import setting


class Model(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"


MODEL_INFO = {
    Model.GPT_3_5_TURBO: {
        "provider": "OpenAI",
        "unit_price_input": 0.0015,  # for 1000 tokens
        "unit_price_output": 0.0002,  # for 1000 tokens
        "extra_tokens_per_message": 3,  # openai add 3 extra tokens to every message to format it
        "extra_tokens_for_reply": 3,  # every reply is primed with <|start|>assistant<|message|>, hence the 3 here
    }
}

DEFAULT_PROMPTS = {
    "REVISE_FOR_SEARCH": "Revise the following text in its own language to create an effective Google search query. "
                         "Be sure to include specific details or criteria to refine the search and find the most relevant results. "
                         "Output nothing but the revised query. The text is:\n"
}


class LLMTrigger(BaseTrigger):
    def __init__(
            self,
            content: str = "",
            stream: bool = True,
            model_name=Model.GPT_3_5_TURBO,
            conversation_id: str = "",
            temperature: float = 0.5,
    ):
        super().__init__(content=content)  # input to model
        self.model_name = model_name
        self.stream = stream
        self.conversation_id = conversation_id
        self.history = []  # history of conversation
        self.temperature = temperature

    def to_dict(self):
        return {
            "content": self.content,
            "stream": self.stream,
            "prompt": {} if self.prompt is None else self.prompt.to_dict(),
            "model_name": self.model_name,
            "conversation_id": self.conversation_id,
            "temperature": self.temperature,
            "history": self.history,
        }


class LLMResult(BaseResult):
    def __init__(
            self,
            trigger: LLMTrigger,
            content: str = "",
            success: bool = True,
            error: Optional[Error] = None,
            error_message: str = "",
    ):
        super().__init__(trigger=trigger, content=content, success=success, error=error, error_message=error_message)
        self.input_token_usage = 0
        self.output_token_usage = 0

    @property
    def cost(self) -> float:
        return (
                self.input_token_usage / 1000 * MODEL_INFO[self.trigger.model_name]["unit_price_input"]
                + self.output_token_usage / 1000 * MODEL_INFO[self.trigger.model_name]["unit_price_output"]
        )

    def to_dict(self):
        return {
            "trigger": self.trigger.to_dict() if self.trigger is not None else {},
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "error_message": self.error_message,
            "input_token_usage": self.input_token_usage,
            "output_token_usage": self.output_token_usage,
            "cost": self.cost,
        }


class LLMAgent(BaseAgent):
    import openai

    TRIGGER_CLASS = LLMTrigger
    RESULT_CLASS = LLMResult

    def warm_up(self, trigger_attrs: Dict):
        if trigger_attrs.get("prompt"):
            prompt = trigger_attrs["prompt"] + "\n" + trigger_attrs["user_input"]
        elif trigger_attrs.get("prompt_name"):
            prompt = DEFAULT_PROMPTS[trigger_attrs["prompt_name"]] + "\n" + trigger_attrs["user_input"]
        else:
            prompt = trigger_attrs["user_input"]
        trigger = self.TRIGGER_CLASS(content=prompt, stream=trigger_attrs.get("stream", True))
        return trigger, self.RESULT_CLASS(trigger=trigger)

    def do(self, trigger: LLMTrigger, result: LLMResult):
        self.openai.api_key = setting.get("OPENAI_API_KEY")
        self.openai.proxy = setting.get("PROXY")
        if trigger.stream:
            return self.stream_chat(trigger=trigger, result=result)
        else:
            return self.chat(trigger=trigger, result=result)

    def stream_chat(self, trigger: LLMTrigger, result: LLMResult, cutoff_value=6) -> Iterator[str | LLMResult]:
        token_encoding = tiktoken.encoding_for_model(trigger.model_name)
        complete_message = ""
        collected_message = ""
        try:
            res = self.openai.ChatCompletion.create(
                model=trigger.model_name,
                messages=trigger.history + [{"role": "user", "content": trigger.content}],
                temperature=trigger.temperature,
                stream=trigger.stream,
            )
            for chunk in res:
                collected_message += chunk["choices"][0]["delta"].get("content", "")  # extract the message
                if len(token_encoding.encode(collected_message)) >= cutoff_value:
                    complete_message += collected_message
                    collected_message = ""
                    value = yield complete_message
                    if value == "STOP":
                        res.close()
                        break
            if collected_message:
                complete_message += collected_message
                yield complete_message
            input_token_usage, output_token_usage = self._calculate_token_usages(
                encoding=token_encoding,
                model_name=trigger.model_name,
                history_messages=trigger.history + [{"role": "user", "content": trigger.content}],
                reply_message=complete_message,
            )
            result.set(
                content=complete_message,
                input_token_usage=input_token_usage,
                output_token_usage=output_token_usage,
            )
            logger.debug(f"Stream chat completed with message: {complete_message}")
        except Exception as e:
            result.set(success=False, error=Error.API_CONNECTION,
                       error_message=QTranslator.tr("Connection to API failed."))
            logger.error(f"Connection to API failed: {e}")
        yield result

    def chat(self, trigger: LLMTrigger, result: LLMResult):
        token_encoding = tiktoken.encoding_for_model(trigger.model_name)
        try:
            res = self.openai.ChatCompletion.create(
                model=trigger.model_name,
                messages=trigger.history + [{"role": "user", "content": trigger.content}],
                temperature=trigger.temperature,
                stream=trigger.stream,
            )
            message = res["choices"][0].message.content
            input_token_usage, output_token_usage = self._calculate_token_usages(
                encoding=token_encoding,
                model_name=trigger.model_name,
                history_messages=trigger.history + [{"role": "user", "content": trigger.content}],
                reply_message=message,
            )
            result.set(
                content=message,
                input_token_usage=input_token_usage,
                output_token_usage=output_token_usage,
            )
        except Exception as e:
            result.set(success=False, error=Error.API_CONNECTION,
                       error_message=QTranslator.tr("Connection to API failed."))
        return result

    @staticmethod
    def _calculate_token_usages(encoding, model_name, history_messages, reply_message) -> Tuple[int, int]:
        """see https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        for reference
        """
        extra_tokens_per_message = MODEL_INFO[model_name].get("extra_tokens_per_message", 0)
        input_num_tokens = 0
        for message in history_messages:
            input_num_tokens += extra_tokens_per_message
            for key, value in message.items():
                input_num_tokens += len(encoding.encode(value))
                if key == "name":
                    input_num_tokens += 1
        input_num_tokens += MODEL_INFO[model_name].get("extra_tokens_for_reply", 0)
        output_num_tokens = len(encoding.encode(reply_message))
        return input_num_tokens, output_num_tokens


if __name__ == "__main__":
    agent = LLMAgent()
    trigger = LLMTrigger(content="count from 1 to 15")
    chat_response = agent.stream_chat(trigger=trigger, result=LLMResult(trigger=trigger))
    while True:
        try:
            chunk = next(chat_response)
            print(chunk, "\n")
            if isinstance(chunk, str):
                if "18" in chunk:
                    chunk = chat_response.send("STOP")
                    assert isinstance(chunk, LLMResult)
                    print("LLMResullt：", chunk.content)
                    print("token usage:", chunk.input_token_usage, chunk.output_token_usage)
                    break
        except:
            print(type(chunk))
            break

from typing import (
    List,
    Dict,
    Any,
    Optional,
)
from pydantic import BaseModel
from litellm.utils import (
    ModelResponse,
    Usage,
    Message,
    Choices,
    StreamingChoices,
    Delta,
    FunctionCall,
    Function,
    ChatCompletionMessageToolCall,
)
from openai._models import BaseModel as OpenAIObject
from openai.types.chat.chat_completion import *
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


class PMDetail(BaseModel):
    model: str
    name: str
    version_uuid: str
    version: int
    log_uuid: str


class LLMResponse(OpenAIObject):
    api_response: Optional[ModelResponse] = None
    raw_output: Optional[str] = None
    parsed_outputs: Optional[Dict[str, Any]] = None
    error: Optional[bool] = None
    error_log: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    pm_detail: Optional[PMDetail] = None


class LLMStreamResponse(OpenAIObject):
    api_response: Optional[ModelResponse] = None
    raw_output: Optional[str] = None
    parsed_outputs: Optional[Dict[str, Any]] = None
    error: Optional[bool] = None
    error_log: Optional[str] = None
    function_call: Optional[ChoiceDeltaFunctionCall] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
    pm_detail: Optional[PMDetail] = None


class FunctionModelConfig(BaseModel):
    """Response Class for FunctionModel.get_config()
    prompts: List[Dict[str, Any]] = []
        each prompt can have role, content, name, function_call, and tool_calls
    version_detail: Dict[str, Any] = {}
        version_detail has "model", "uuid", "parsing_type" and "output_keys".
    model: str
        model name (e.g. "gpt-3.5-turbo")
    name: str
        name of the FunctionModel.
    version_uuid: str
        version uuid of the FunctionModel.
    version: int
        version id of the FunctionModel.
    parsing_type: Optional[str] = None
        parsing type of the FunctionModel.
    output_keys: Optional[List[str]] = None
        output keys of the FunctionModel.
    """

    prompts: List[Dict[str, Any]]
    model: str
    name: str
    version_uuid: str
    version: int
    parsing_type: Optional[str] = None
    output_keys: Optional[List[str]] = None


class PromptModelConfig(FunctionModelConfig):
    """Deprecated. Use FunctionModelConfig instead."""


class ChatModelConfig(BaseModel):
    system_prompt: str
    model: str
    name: str
    version_uuid: str
    version: int
    message_logs: Optional[List[Dict]] = []


class FunctionSchema(BaseModel):
    """
    {
            "name": str,
            "description": Optional[str],
            "parameters": {
                "type": "object",
                "properties": {
                    "argument_name": {
                        "type": str,
                        "description": Optional[str],
                        "enum": Optional[List[str]]
                    },
                },
                "required": Optional[List[str]],
            },
        }
    """

    class _Parameters(BaseModel):
        class _Properties(BaseModel):
            type: str
            description: Optional[str] = ""
            enum: Optional[List[str]] = []

        type: str = "object"
        properties: Dict[str, _Properties] = {}
        required: Optional[List[str]] = []

    name: str
    description: Optional[str] = None
    parameters: _Parameters

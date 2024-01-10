from dataclasses import dataclass
from typing import Any, Callable, Optional

from openai.types.beta.assistant_create_params import ToolAssistantToolsFunction
from openai.types.shared_params import FunctionDefinition, FunctionParameters


@dataclass
class ParameterSpec:
    name: str
    description: str
    required: bool
    default: Optional[str] = None
    enum: Optional[list[str]] = None

    def dict(self) -> dict[str, Any]:
        return {
            "type": "string",  # At this time, no other types are supported.
            "description": self.description or "None",
            "default": self.default or "None",
        }


@dataclass
class FunctionSpec:
    name: str
    description: str
    parameters: list[ParameterSpec]
    action: Callable[..., Any]

    def tool(self) -> ToolAssistantToolsFunction:
        return ToolAssistantToolsFunction(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description or "None",
                parameters=self.json_parameters(),
            ),
        )

    def json_parameters(self) -> FunctionParameters:
        # For some reason OpenAI doesn't continue to type this, but instead just provides dict[str, object].
        # In any case, it's supposed to be a JSONSchema object, so we'll just do that manually for now.
        # https://github.com/openai/openai-python/blob/main/src/openai/types/shared_params/function_parameters.py
        parameters = {
            "type": "object",
            "properties": {param.name: param.dict() for param in self.parameters},
            "required": [param.name for param in self.parameters if param.required],
        }

        # enum processing - do this in a second pass to avoid empty enums
        for param in self.parameters:
            if param.enum:
                parameters["properties"][param.name]["enum"] = list(param.enum)

        return parameters


@dataclass
class FunctionCall:
    call_id: str
    function: FunctionSpec
    parameters: dict[str, Any]

    def dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "function": self.function.name,
            "parameters": self.parameters,
        }


@dataclass
class FunctionResult:
    call: FunctionCall
    return_value: Any  # See note above on parametric return types
    stdout: str

    def dict(self) -> dict:
        return {
            "call_id": self.call.call_id,
            "function": self.call.function.name,
            "return_value": self.return_value,
            "stdout": self.stdout,
        }

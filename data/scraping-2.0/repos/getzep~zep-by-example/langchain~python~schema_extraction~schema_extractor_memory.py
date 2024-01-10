from __future__ import annotations

from abc import ABC
from typing import Any, Dict

from langchain.chains import create_extraction_chain_pydantic
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel


class ExtractorSchema(BaseModel, ABC):
    """Base class for extractor models."""

    class Config:
        exclude_unset = True

    def recursive_update(self, model_new: ExtractorSchema | None) -> None:
        """Recursively update model values."""
        if model_new is None:
            return

        for name, field in self.__annotations__.items():
            new_field_value = getattr(model_new, name, None)
            old_field_value = getattr(self, name)

            if isinstance(old_field_value, ExtractorSchema):
                if new_field_value is not None and new_field_value != {}:
                    old_field_value.recursive_update(new_field_value)
            else:
                if new_field_value is not None and new_field_value != "":
                    setattr(self, name, new_field_value)


class SchemaExtractorMemory(ConversationBufferMemory):
    """Memory for extracting values from chat message inputs."""

    model: ExtractorSchema | None = None
    llm: ChatOpenAI | None = None
    chain: Chain | None = None

    def __init__(self, model_schema: ExtractorSchema, llm: ChatOpenAI, **kwargs: Any):
        super().__init__(**kwargs)

        self.model = model_schema
        self.llm = llm
        self.chain = create_extraction_chain_pydantic(
            pydantic_schema=type(model_schema), llm=llm
        )

    def _merge_model_values(
        self, model_values_new: ExtractorSchema, model_values_old: ExtractorSchema
    ) -> ExtractorSchema | None:
        """Merge new model values into old model values."""
        if model_values_old is not None and model_values_new is not None:
            model_values_old.recursive_update(model_values_new)

        return model_values_old

    def _get_first_model_value(
        self, model_values: list[ExtractorSchema] | ExtractorSchema
    ) -> ExtractorSchema | None:
        """Get first model value from list of model values."""
        if isinstance(model_values, list):  # Added check for list instance
            return (
                None
                if model_values is None or len(model_values) == 0
                else model_values[0]
            )
        else:
            return model_values

    def _extract_model_values(self, input_str: str) -> None:
        """Extract values from inputs to be used in model."""
        model_values = self.chain.run(input_str)

        model_values = self._get_first_model_value(model_values)  # type: ignore

        self.model = self._merge_model_values(model_values, self.model)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)

        self._extract_model_values(input_str)

        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

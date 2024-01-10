from __future__ import annotations
from pydantic.v1 import BaseModel
from langchain.schema import BaseOutputParser
from typing import List, Dict, Any, Callable
from json import JSONDecodeError
import json
import re
line_template = '\t"{name}": {type}  // {description}'

def _costum_json_parser(json_str):
        match =  re.search(r"```(json)?(.*)```", json_str, re.DOTALL)

        if match is None:
            json_str = json_str
        else:
        # If match found, use the content within the backticks
            json_str = match.group(2)
        json_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '').replace('\"', '"')
        result = json.loads(json_str)
        return result

class ResponseInstruct(BaseModel):
    """A schema for a response from a structured output parser."""
    name: str
    """The name of the schema."""
    description: str
    """The description of the schema."""
    type: str = "string"
    """The type of the response."""

def _get_sub_string(schema: ResponseInstruct) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type=schema.type
    )

STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
[{{
{format}
}},
{{
{format2}
}}]
```"""
class OutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a structured output."""

    response_schemas: List[ResponseInstruct]
    """The schemas for the response."""

    @classmethod
    def from_response_schemas(
        cls, response_schemas: List[ResponseInstruct]
    ) -> OutputParser:
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self) -> str:
        """Get the format instructions for the output of the LLM call.

        output:
        # The output should be a Markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }
        # ```
    
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
 
        
        return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str, format2=schema_str)

    def parse(self, text: str) -> Any:
        expected_keys = [rs.name for rs in self.response_schemas]
        return _costum_json_parser(text)

    @property
    def _type(self) -> str:
        return "structured"
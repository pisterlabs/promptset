import json
from typing import Type, TypeVar, Callable, Generic
from pydantic import BaseModel

from learning.llm.llm import Llm

from anthropic import HUMAN_PROMPT, AI_PROMPT


_Parameters = TypeVar("_Parameters", bound=BaseModel)


class SchemaTool(BaseModel, Generic[_Parameters]):
    parameters: Type[_Parameters]
    evaluate: Callable[[_Parameters], str]

    description: str = None
    name: str = None

    template: str = None

    def run(self, llm: Llm, context: str) -> str:
        template = self.template or _DEFAULT_TEMPLATE

        prompt = build_pydantic_tool_prompt(
            tool_name=self.name,
            description=self.description,
            response_model=self.parameters,
            context=context,
            template=template,
        )
        response = llm.complete(prompt)
        # todo retry validation?
        parameters = self.parameters.model_validate_json(response)
        return self.evaluate(parameters)


_DEFAULT_TEMPLATE = f"""{HUMAN_PROMPT} \
You an AI assistant using the "{{tool_name}}" tool, this is it's description: {{description}}

Your answer should be a valid JSON object that follows this schema:
{{json_schema}}

The situation you are using the tool in is: {{context}}

Please respond directly with an instance of the above schema, include no additional text.

{AI_PROMPT} {{{{\
"""


def build_pydantic_tool_prompt(
    tool_name: str,
    description: str,
    response_model: BaseModel,
    context: str,
    template: str = None,
) -> str:
    template = template or _DEFAULT_TEMPLATE
    json_schema = json.dumps(response_model.model_json_schema(), indent=2)

    return _DEFAULT_TEMPLATE.format(
        tool_name=tool_name,
        description=description,
        json_schema=json_schema,
        context=context,
    )

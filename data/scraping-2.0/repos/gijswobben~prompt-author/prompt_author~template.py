from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Type, TypeVar

from langchain import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel

from prompt_author.personas.default import DefaultPersona
from prompt_author.personas.persona import Persona

T = TypeVar("T", bound=BaseModel)


class TemplateRegistry:
    _instance: TemplateRegistry | None = None
    templates: dict[str, type[Template]] = {}

    def __new__(
        cls: Type[TemplateRegistry], *args: Any, **kwargs: Any
    ) -> TemplateRegistry:
        if cls._instance is None:
            cls._instance = super(TemplateRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, template: type[Template]) -> None:
        self.templates[template.name] = template

    def get(self, template: str) -> type[Template]:
        template_class = self.templates.get(template, None)
        if template_class is None:
            raise Exception(f"Template {template} not found")
        return template_class


class Template:
    name: str

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        # Get the registry and register the template. The registry is a
        # singleton, so it will be the same instance for every call.
        TemplateRegistry(cls).register(cls)

    def __init__(
        self,
        llm: BaseChatModel,
        persona: Persona | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.persona = persona if persona is not None else DefaultPersona()
        self.llm = llm
        self.memory: dict[str, Any] = {
            **kwargs,
            "persona": self.persona.to_string(),
        }
        self.verbose = verbose

    def load_prompt(self, prompt_file: Path) -> str:
        return prompt_file.read_text().strip()

    def step(
        self,
        prompt_file: Path,
        model: Type[T] | None = None,
        prompt_params: dict[str, Any] | None = None,
    ) -> T:
        """Use a prompt to generate data in a particular format
        (Pydanitc model).

        Args:
            prompt_file (Path): The name of the prompt file to use.
            model (Type[T] | None): The Pydantic model to use to parse the
                response. If None, the response will not be parsed.
            prompt_params (dict[str, Any] | None, optional): The
                parameters to use for filling the prompt. Defaults to
                None.

        Raises:
            Exception: Raised when the response cannot be parsed.

        Returns:
            T: Pydantic model (same as `model` input)
        """

        # Define the output parser and wrap it to automatically fix
        # errors
        if model is not None:
            parser = PydanticOutputParser(pydantic_object=model)  # type: ignore
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)

        # Load the prompt
        prompt_text = self.load_prompt(prompt_file=prompt_file)
        if model is not None:
            prompt_text += "\n\n{format_instructions}"

        # Extract all input variables (wrapped in { } from the prompt
        # text) using regex
        pattern = r"{\s*(?P<variable>\w+)\s*}"
        input_variables = re.findall(pattern, prompt_text)

        # Define the prompt template
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=input_variables,
        )

        # Combine the arguments for the prompt
        arguments = {
            **self.memory,
            **(prompt_params if prompt_params is not None else {}),
            "format_instructions": fixing_parser.get_format_instructions()
            if model is not None
            else "",
        }

        # Format the prompt
        formatted_prompt = prompt.format_prompt(
            **{key: value for key, value in arguments.items() if key in input_variables}
        )

        # Get a response
        response = self.llm.predict(text=formatted_prompt.to_string())

        # Parse the response, retry with the fixer if needed
        if model is not None:
            response = fixing_parser.parse(response)
        return response

    def run(self) -> Any:
        raise NotImplementedError()

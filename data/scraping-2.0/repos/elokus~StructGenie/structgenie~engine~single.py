import uuid
from typing import Union, Callable, Type

from colorama import Fore
from pydantic import BaseModel, Field

from structgenie.base import BasePromptBuilder, BaseValidator, BaseGenerationDriver
from structgenie.components import ExampleSelector
from structgenie.components import (
    load_input_schema,
    init_input_schema,
    OutputModel,
    load_output_model,
    init_output_model
)
from structgenie.driver import LangchainDriverBasic
from structgenie.errors import ParsingError, ValidationError
from structgenie.utils.operator.default import parse_default
from structgenie.utils.parsing import (
    parse_yaml_string,
    dump_to_yaml_string,
    format_inputs,
    prepare_inputs_placeholders
)
from structgenie.utils.templates import (
    extract_sections, load_default_template, load_system_config
)


def run_output_fixing_parser(text: str, output_model: OutputModel):
    from colorama import Fore
    print(Fore.RED + "Fixing parsing error for output:\n" + Fore.RESET)
    print(Fore.MAGENTA + text + Fore.RESET)
    fixing_engine = StructGenie.from_defaults("fix_parsing_error", output_model=output_model)
    fixing_engine.output_fixing_parser = None
    output = fixing_engine.run(inputs=dict(last_output=text))
    print(Fore.GREEN + "Parsing error fixed! With output:" + Fore.RESET)
    print(Fore.GREEN + str(output) + Fore.RESET)
    return output


class StructGenie(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4().hex))

    # executor
    driver: Type[BaseGenerationDriver] = LangchainDriverBasic

    # prompt
    prompt_builder: BasePromptBuilder = None

    # validation settings
    validator: BaseValidator = None

    # parser
    output_parser: Callable = parse_yaml_string
    output_fixing_parser: Union[Callable, None] = run_output_fixing_parser

    # run settings
    max_retries: int = 4
    input_schema: str = None
    output_model: OutputModel = None
    examples: ExampleSelector = None
    instruction: str = None
    debug: bool = False

    # run states
    last_error: Union[Exception, None] = None
    partial_variables: dict = None

    return_reasoning: bool = False

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_defaults(cls, template_name: str, **kwargs):
        template = load_default_template(template_name)
        return cls.from_template(template, **kwargs)

    @classmethod
    def from_template(
            cls,
            schema_template: str,
            output_model: Union[OutputModel, dict, str] = None,
            prompt_kwargs: dict = None,
            partial_output_model: dict[str, Union[OutputModel, dict, str]] = None,
            **kwargs):
        """Build Prediction Engine from template."""
        sections = extract_sections(schema_template.strip())

        system_config = sections.get("system_config")
        instruction = sections.get("instruction")
        examples = ExampleSelector.load_examples(sections.get("examples"))
        output_model = load_output_model(
            sections, examples, output_model=output_model, partial_output_model=partial_output_model
        )
        input_schema = load_input_schema(sections, examples)

        return cls.load_engine(
            instruction=instruction,
            input_schema=input_schema,
            output_model=output_model,
            examples=examples,
            prompt_kwargs=prompt_kwargs,
            system_config=system_config,
            **kwargs
        )

    @classmethod
    def load_examples(cls, examples: Union[str, list] = None):
        """Load examples from a string, list or file."""
        return ExampleSelector.load_examples(examples)

    @classmethod
    def from_instruction(
            cls,
            instruction: str,
            input_schema: Union[str, dict, list] = None,
            output_model: Union[OutputModel, dict, str] = None,
            examples: Union[str, list] = None,
            prompt_kwargs: dict = None,
            partial_output_model: dict[str, Union[OutputModel, dict, str]] = None,
            **kwargs) -> "StructGenie":
        """Build Prediction Engine from instructions."""
        if examples:
            examples = ExampleSelector.load_examples(examples)

        output_model = init_output_model(output_model, examples=examples, partial_output_model=partial_output_model)
        input_schema = init_input_schema(input_schema, examples=examples)

        return cls.load_engine(
            instruction=instruction,
            input_schema=input_schema,
            output_model=output_model,
            examples=examples,
            prompt_kwargs=prompt_kwargs,
            **kwargs
        )

    @classmethod
    def load_engine(
            cls,
            instruction: str,
            input_schema: str,
            output_model: OutputModel,
            examples: ExampleSelector = None,
            prompt_kwargs: dict = None,
            system_config: Union[dict, str] = None,
            **kwargs) -> "StructGenie":

        from structgenie.components.prompt.builder import PromptBuilder
        from structgenie.components.validation import Validator

        kwargs = kwargs or {}

        system_config = load_system_config(system_config)
        if system_config["engine"]:
            _kwargs = system_config
            _kwargs.update(kwargs)
            kwargs = _kwargs

        if system_config["partial_variables"]:
            if kwargs.get("partial_variables"):
                partial_variables = system_config["partial_variables"]
                partial_variables.update(kwargs["partial_variables"])
                kwargs["partial_variables"] = partial_variables
            else:
                kwargs["partial_variables"] = system_config["partial_variables"]

        prompt_kwargs = prompt_kwargs or {}
        prompt_builder = PromptBuilder(
            instruction=instruction,
            examples=examples,
            output_model=output_model,
            input_schema=input_schema,
            **prompt_kwargs
        )
        validator = Validator.from_output_model(output_model)

        return cls(
            instruction=instruction,
            prompt_builder=prompt_builder,
            validator=validator,
            output_model=output_model,
            input_schema=input_schema,
            examples=examples,
            **kwargs
        )

    # === Setters ===

    def set_example_selector(self, examples: ExampleSelector):
        self.prompt_builder.examples = examples
        self.examples = examples

    def set_instruction(self, instruction: str):
        self.prompt_builder.instruction = instruction
        self.instruction = instruction

    def set_output_model(self, output_model: OutputModel):
        from structgenie.components.validation import Validator
        self.prompt_builder.output_model = output_model
        self.validator = Validator.from_output_model(output_model)
        self.output_model = output_model

    # === Run ===

    def run(self, inputs: dict, **kwargs):
        """Run the chain.

        Args:
            inputs (dict): The inputs for the chain.
            **kwargs: Keyword arguments for the chain.

        Returns:
            Any: The output of the chain.
        """
        self.last_error = None

        n_run = 0
        while n_run <= self.max_retries:
            try:
                return self._run(inputs, self.last_error, **kwargs)
            except Exception as e:
                print(f"Error: {e}")
                self.last_error = e
                n_run += 1
                if self.debug:
                    raise e

    def _run(self, inputs: dict, error: Exception, **kwargs):
        """Run the chain.

        Args:
            inputs (dict): The inputs for the chain.
            error (Exception): The error of the previous run.
            **kwargs: Keyword arguments for the chain.

        Returns:
            Any: The output of the chain.
        """

        # prepare
        inputs = self.prep_inputs(inputs, **kwargs)
        prompt = self.prep_prompt(error, **inputs)
        inputs_ = self.format_inputs(prompt, inputs, **kwargs)
        executor = self.prep_executor(prompt, **kwargs)

        if self.debug:
            print(prompt.format(**inputs_))

        # generate
        text = self._call_executor(executor, inputs_)

        if self.debug:
            print(">>> Generated text:\n")
            print(text)
        # parse
        output = self.parse_output(text, inputs)
        # validate
        self.validate_output(output, inputs)

        return output

    def prep_prompt(self, error: Exception = None, **kwargs) -> str:
        """Prepare the prompt for the chain.

        Args:
            error (Exception): The error message.
            **kwargs: Keyword arguments for the prompt.

        Returns:
            str: The prompt.
        """
        if error is None:
            return self.prompt_builder.build(**kwargs)

        if isinstance(error, ParsingError):
            return self.prompt_builder.fix_parsing(error=str(error), **kwargs)

        if isinstance(error, ValidationError):
            return self.prompt_builder.fix_validation(error=str(error), **kwargs)

        return self.prompt_builder.build(**kwargs)

    def prep_executor(self, prompt: str, **kwargs) -> BaseGenerationDriver:
        """Prepare the executor for the chain.

        Args:
            prompt (str): The prompt for the chain.
            **kwargs: Keyword arguments for the executor.

        Returns:
            Any: The executor.
        """
        return self.driver.load_driver(prompt=prompt, **kwargs)

    def prep_inputs(self, inputs: dict, **kwargs) -> dict:
        """Analyzes input variables in prompt and prepares inputs for executor."""
        if self.partial_variables:
            inputs.update(self.partial_variables)
        return inputs

    def format_inputs(self, prompt: str, inputs: dict, **kwargs) -> dict:
        """Analyzes input variables in prompt and prepares inputs for executor."""
        prompt, placeholder_map = prepare_inputs_placeholders(prompt, inputs, **kwargs)
        return format_inputs(placeholder_map, self.input_schema)

    # === output parsing ===

    def parse_output(self, text: str, inputs: dict) -> dict:
        """Parse the outputs of the chain.

        Args:
            text (str): The output of the chain.

        Returns:
            Dict: The parsed output.

        Raises:
            ParsingError: If the output could not be parsed.
        """
        try:
            output = self.output_parser(text)
        except Exception as e:
            print(Fore.RED + f"Error while parsing output: {e}" + Fore.RESET)
            output = self._parse_output(text, e)
        output = self._prefix_output(output)
        output = self._parse_defaults(output, inputs)
        return output

    def _prefix_output(self, output: any) -> dict:
        """Prefix the output with the output prefix if defined."""
        if len(self.output_model) == 1:
            if len(output) > 1 or self.output_model.keys()[0] not in output:
                return {self.output_model.keys()[0]: output}
        return output

    def _parse_output(self, text: str, error: Exception) -> dict:
        """Run output fixing parser if defined."""

        if self.output_fixing_parser is None:
            raise ParsingError(f"Error while parsing output: {error}", text)

        try:
            return self.output_fixing_parser(text, output_model=self.output_model)

        except Exception as e:
            raise ParsingError(
                f"Could neither parse nor fix parsing error. Parsing error: {error}. Fixing error: {e}",
                text
            )

    def _parse_defaults(self, output: dict, inputs):
        return parse_default(output, self.output_model, **inputs)

    # === output validation ===

    def validate_output(self, output: dict, formatted_inputs: dict):
        """Validate the output of the chain.

        Args:
            output (Any): The output of the chain.

        Returns:
            Any: The output of the chain.
        """

        validation_errors = self.validator.validate(output, formatted_inputs)
        if validation_errors:
            raise ValidationError(f"Validation failed with errors:\n{validation_errors}", output)

    @staticmethod
    def _call_executor(executor: BaseGenerationDriver, inputs: dict) -> str:
        """Call the executor.

        Args:
            executor (Any): The executor.
            inputs (dict): The inputs for the executor.

        Returns:
            str: The output of the executor.
        """
        return executor.predict(**inputs)

    @staticmethod
    def _prep_inputs(inputs: dict, input_keys: list) -> str:
        """Prepares inputs for executor."""
        return dump_to_yaml_string({k: inputs[k] for k in input_keys if k in inputs})

    def _remove_cot(self, output: dict):
        cot = []
        for key in ["reason", "chain-of-thoughts", "reasoning"]:
            if key in output:
                value = output.pop(key)
                if isinstance(value, list):
                    value = "\n".join(value)
                cot.append(value)
        if cot:
            self.cot = cot[0]
        return output

    @property
    def execution_type(self):
        return "sync"

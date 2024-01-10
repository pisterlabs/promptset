from enum import Enum
from inspect import isclass
from typing import Callable, Literal, Optional

from docstring_parser import Docstring, DocstringParam
from pydantic.fields import ModelField
from pydantic.main import BaseModel
from typing_extensions import Literal as ExtensionLiteral

from openai_func_call.function_parsing import create_model_for_func_params, get_viable_docstring


class CallableFunction(BaseModel):
    name: str
    # alias: str # TODO: Add alias to support non-unique function names
    params_model: type[BaseModel]
    function: Callable
    api_dict: dict
    doc_str: Docstring

    class Config:
        arbitrary_types_allowed = True


def func_to_api_dict(func: Callable) -> dict:
    """Convert a function to a JSON item for the OpenAI API."""
    return func_to_callable_function(func).api_dict


def func_to_callable_function(func: Callable) -> CallableFunction:
    """Convert a function to a CallableFunction object."""
    func_docstring = get_viable_docstring(func)
    description = get_full_description(func_docstring)
    pydantic_model = create_model_for_func_params(func)

    # check
    param_names = [param.arg_name for param in func_docstring.params]
    assert set(pydantic_model.__fields__.keys()) == set(param_names), "Parsed model and docstring do not match."

    # properties
    properties = create_properties_field(pydantic_model, func_docstring)

    # required field
    required = get_required_field(pydantic_model)

    # assembly
    api_dict = {
        "name": func.__name__,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    return CallableFunction(
        name=func.__name__,
        params_model=pydantic_model,
        function=func,
        api_dict=api_dict,
        doc_str=func_docstring,
    )


def get_required_field(pydantic_model: BaseModel) -> list[str]:
    return [name for name, value in pydantic_model.__fields__.items() if value.required]


def create_properties_field(pydantic_model: BaseModel, func_docstring: Docstring) -> dict:
    properties = {}
    docstring_params_dict = {param.arg_name: param for param in func_docstring.params}
    for name in pydantic_model.__fields__.keys():
        properties[name] = convert_to_property(pydantic_model.__fields__[name], docstring_params_dict[name])
    return properties


def get_full_description(func_docstring: Docstring) -> str:
    description = func_docstring.short_description
    if func_docstring.long_description:
        description += " " + func_docstring.long_description
    return description


def convert_to_property(field: ModelField, param: DocstringParam) -> dict:
    return_dict = {"description": param.description}
    # Note, field.type_ ignores Optional, which is what we want
    if field.type_ == str:
        return_dict["type"] = "string"
    elif field.type_ == int:
        return_dict["type"] = "integer"
    elif field.type_ == float:
        return_dict["type"] = "number"  # TODO: Check if this works
    elif field.type_ == bool:
        return_dict["type"] = "boolean"  # TODO: Check if this works
    elif isclass(field.type_) and issubclass(field.type_, Enum) and issubclass(field.type_, str):
        # TODO: non-string enums
        return_dict["type"] = "string"
        return_dict["enum"] = [enum.value for enum in field.type_]
    elif (
        hasattr(field.type_, "__args__")
        and hasattr(field.type_, "__origin__")
        and field.type_.__origin__ in [Literal, ExtensionLiteral]
    ):
        # get literal values
        return_dict["type"] = "string"
        return_dict["enum"] = list(field.type_.__args__)
    else:
        raise ValueError(f"Type {field.type_} is not supported.")

    return return_dict


class SomeEnum(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"


if __name__ == "__main__":
    # Example function with arguments
    def example_func(
        name: str,
        age: int = 30,
        email: str = "",
        aoeu: Optional[str] = None,
        enumting: SomeEnum = SomeEnum.A,
        literalting: Literal["a", "b", "c"] = "a",
        literaltingtwo: ExtensionLiteral[1, 2, 3] = 1,
    ):
        """This function does stuff.

        :param name: Name of person
        :param age: Age of person
        :param email: Email of person
        :param aoeu: Aoeu is irrelevant
        :param enumting: Some enum
        :param literalting: Some literal
        :param literaltingtwo: Something literal
        """
        pass

    # Create Pydantic model from function arguments
    result = func_to_api_dict(example_func)

    print(result)

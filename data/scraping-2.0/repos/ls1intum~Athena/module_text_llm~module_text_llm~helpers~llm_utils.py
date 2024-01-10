from typing import Optional, Type, TypeVar, List
from pydantic import BaseModel, ValidationError
import tiktoken

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from athena import emit_meta, get_experiment_environment

T = TypeVar("T", bound=BaseModel)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_prompt(chat_prompt: ChatPromptTemplate, prompt_input: dict) -> int:
    """Returns the number of tokens in a chat prompt."""
    return num_tokens_from_string(chat_prompt.format(**prompt_input))


def check_prompt_length_and_omit_features_if_necessary(prompt: ChatPromptTemplate, 
                                                       prompt_input: dict, 
                                                       max_input_tokens: int, 
                                                       omittable_features: List[str],
                                                       debug: bool):
    """Check if the input is too long and omit features if necessary.

    Note: Omitted features will be replaced with "omitted" in the prompt

    Args:
        prompt (ChatPromptTemplate): Prompt template
        prompt_input (dict): Prompt input
        max_input_tokens (int): Maximum number of tokens allowed
        omittable_features (List[str]): List of features that can be omitted, ordered by priority (least important first)
        debug (bool): Debug flag

    Returns:
        (dict, bool): Tuple of (prompt_input, should_run) where prompt_input is the input with omitted features and 
                      should_run is True if the model should run, False otherwise
    """
    if num_tokens_from_prompt(prompt, prompt_input) <= max_input_tokens:
        # Full prompt fits into LLM context => should run with full prompt
        return prompt_input, True

    omitted_features = []

    # Omit features until the input is short enough
    for feature in omittable_features:
        if feature in prompt_input:
            omitted_features.append(feature)
            prompt_input[feature] = "omitted"
            if num_tokens_from_prompt(prompt, prompt_input) <= max_input_tokens:
                if debug:
                    emit_meta("omitted_features", omitted_features)
                return prompt_input, True

    # If we get here, we couldn't omit enough features
    return prompt_input, False


def supports_function_calling(model: BaseLanguageModel):
    """Returns True if the model supports function calling, False otherwise

    Args:
        model (BaseLanguageModel): The model to check

    Returns:
        boolean: True if the model supports function calling, False otherwise
    """
    return isinstance(model, ChatOpenAI)


def get_chat_prompt_with_formatting_instructions(
            model: BaseLanguageModel,
            system_message: str, 
            human_message: str,
            pydantic_object: Type[T]
        ) -> ChatPromptTemplate:
    """Returns a ChatPromptTemplate with formatting instructions (if necessary)

    Note: Does nothing if the model supports function calling

    Args:
        model (BaseLanguageModel): The model to check if it supports function calling
        system_message (str): System message
        human_message (str): Human message
        pydantic_object (Type[T]): Model to parse the output

    Returns:
        ChatPromptTemplate: ChatPromptTemplate with formatting instructions (if necessary)
    """
    if supports_function_calling(model):
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    output_parser = PydanticOutputParser(pydantic_object=pydantic_object)
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message + "\n{format_instructions}")
    system_message_prompt.prompt.partial_variables = {"format_instructions": output_parser.get_format_instructions()}
    system_message_prompt.prompt.input_variables.remove("format_instructions")
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message + "\n\nJSON response following the provided schema:")
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


async def predict_and_parse(
        model: BaseLanguageModel, 
        chat_prompt: ChatPromptTemplate, 
        prompt_input: dict, 
        pydantic_object: Type[T], 
        tags: Optional[List[str]]
    ) -> Optional[T]:
    """Predicts an LLM completion using the model and parses the output using the provided Pydantic model

    Args:
        model (BaseLanguageModel): The model to predict with
        chat_prompt (ChatPromptTemplate): Prompt to use
        prompt_input (dict): Input parameters to use for the prompt
        pydantic_object (Type[T]): Pydantic model to parse the output
        tags (Optional[List[str]]: List of tags to tag the prediction with

    Returns:
        Optional[T]: Parsed output, or None if it could not be parsed
    """
    experiment = get_experiment_environment()

    tags = tags or []
    if experiment.experiment_id is not None:
        tags.append(f"experiment-{experiment.experiment_id}")
    if experiment.module_configuration_id is not None:
        tags.append(f"module-configuration-{experiment.module_configuration_id}")
    if experiment.run_id is not None:
        tags.append(f"run-{experiment.run_id}")

    if supports_function_calling(model):
        chain = create_structured_output_chain(pydantic_object, llm=model, prompt=chat_prompt, tags=tags)
        
        try:
            return await chain.arun(**prompt_input)
        except (OutputParserException, ValidationError):
            # In the future, we should probably have some recovery mechanism here (i.e. fix the output with another prompt)
            return None

    output_parser = PydanticOutputParser(pydantic_object=pydantic_object)
    chain = LLMChain(llm=model, prompt=chat_prompt, output_parser=output_parser, tags=tags)
    try:
        return await chain.arun(**prompt_input)
    except (OutputParserException, ValidationError):
        # In the future, we should probably have some recovery mechanism here (i.e. fix the output with another prompt)
        return None

import asyncio
import csv
from asyncio import sleep
from pathlib import Path
from typing import List, Dict, Iterator, Coroutine, Any

from langchain_community.chat_models import ChatOpenAI

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from openai import RateLimitError

from .chains import build_parameter_chain, build_parameter_value_chain, extract_list_from_response, build_formatter_chain
from .loading import OmniClass
from .typedefs import Parameter

load_dotenv()

SAVE_PATH = Path('../data')
ORDINALS = [
    "first",
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "13th",
    "14th",
    "15th",
    "16th",
    "17th",
    "18th",
    "19th",
    "twentieth"
]

GPT3_LOW_T = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3)
GPT3_HIGH_T = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)


PARAMETER_CHAIN = build_parameter_chain(GPT3_HIGH_T)
VALUE_CHAIN = build_parameter_value_chain(GPT3_LOW_T, GPT3_LOW_T)
FORMATTER_CHAIN = build_formatter_chain(GPT3_LOW_T)


async def _generate_parameters(product_name: str) -> (AIMessage, list[str]):
    llm_response = await PARAMETER_CHAIN.ainvoke({"omniclass": product_name})
    formatted = await FORMATTER_CHAIN.ainvoke({"content": llm_response.content})
    parameter_list = extract_list_from_response(formatted)
    return llm_response, parameter_list


async def generate_parameters(product_name: str) -> (AIMessage, list[str]):
    feedback_msg = f"parameters for {product_name}"
    while True:
        try:
            ai_message, parameters = await _generate_parameters(product_name)
            if len(parameters) == 20:
                return ai_message, parameters
            else:
                print(f"Got less than 20 {feedback_msg}, retrying...")
        except RateLimitError:
            await sleep(15)
        except SyntaxError:
            print(f"Could not understand response when generating {feedback_msg}, retrying...")


def value_coroutines(product_name: str, ai_message: AIMessage,
                     parameters: list[str]) -> list[Coroutine[Any, Any, Parameter]]:
    """ Generate coroutines for generating all values for a given product.

    This is used in `generate_all_values` and in the backend to asynchronously load values.
    """
    assert len(parameters) == len(ORDINALS)

    return [generate_values(product_name, ordinal, ai_message, parameter) for ordinal, parameter in zip(ORDINALS, parameters)]


async def generate_all_values(product_name: str, parameters: list[str], ai_message: AIMessage) -> Dict[str, List[str]]:
    """ Generate all values for a given product in a synchronous manner.

    This is to be used when locally generating a CSV file.
    """
    tasks = value_coroutines(product_name, ai_message, parameters)

    kv_columns = {}

    for parameter in await asyncio.gather(*tasks):
        kv_columns[parameter.name] = parameter.values

    return kv_columns


async def _generate_values(product_name: str, ordinal: str, ai_message: AIMessage) -> list[str]:
    value_response = await VALUE_CHAIN.ainvoke({
        "ordinal": ordinal,
        "ai_message": [ai_message],
        "omniclass": product_name})
    return extract_list_from_response(value_response)


async def generate_values(product_name: str, ordinal: str, ai_message: AIMessage, parameter_name: str) -> Parameter:
    feedback_msg = f"{ordinal} parameter for {product_name}"
    while True:
        try:
            values = await _generate_values(product_name, ordinal, ai_message)
            if len(values) == 20:
                return Parameter(parameter_name, values)
            else:
                print(f"Got less than 20 values for {feedback_msg}, retrying...")
        except RateLimitError:
            await sleep(30)
        except SyntaxError:
            print(f"Could not understand response when generating values for {feedback_msg}, retrying...")


def save_product(path: Path, omniclass: OmniClass, kv_columns: Dict[str, List[str]]) -> None:
    """ Save a product's parameters and values to a CSV file.

    Parameters
    ----------
    path : Path
        The path to save the final CSV file to.
    omniclass : OmniClass
        The OmniClass value. This is used to generate the filename.
    kv_columns : Dict[str, List[str]]
        A dictionary of parameter names to lists of values.

    Returns
    -------
    None
    """
    fn = f'{omniclass.number} {omniclass.name}.csv'
    fn_path = path.joinpath(fn)

    with open(fn_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([i for i in kv_columns.keys()])

        # write values
        for i in range(len(kv_columns.keys())):
            writer.writerow([kv_columns[k][i] for k in kv_columns.keys()])


async def process_product(omniclass: OmniClass):
    omniclass_name = omniclass.name
    print(f"\n*** Processing {omniclass_name}...")
    ai_message, parameters = await generate_parameters(omniclass_name)
    kv_columns = await generate_all_values(omniclass_name, parameters, ai_message)
    save_product(SAVE_PATH, omniclass, kv_columns)
    print(f"\n*** ...Done processing {omniclass_name}. ***\n")

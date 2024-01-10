from typing import Callable

import yaml
from langchain.prompts.example_selector.base import BaseExampleSelector
from taskchain.helper.document_chunker import num_tokens_from_string


def prep_remarks(template: str, error: str = None, **kwargs) -> str:
    remarks = _prep_remarks(error, **kwargs)
    return template.replace("{remarks}", remarks)


def _prep_remarks(error: str = None, **kwargs):
    remarks = ""
    if error:
        remarks = f"\nYour last result returned an error: {error}\nPlease try again.\n"
    if "remarks" in kwargs:
        remarks += "\n{remarks}"
    return remarks


def prep_labels(template: str, labels: list[str]) -> str:
    if "{labels}" not in template:
        return template
    template = template.replace("{labels}", ", ".join(labels))
    return template


def prep_examples(template: str, token_limit: int, selector: BaseExampleSelector, **kwargs) -> str:
    if "{examples}" not in template:
        return template
    if selector is None:
        return template.replace("{examples}", "")
    token_left = token_limit - num_tokens_from_string(template)
    return template.replace("{examples}", selector.select_and_parse_by_input(kwargs, token_left))


def agg_items_by_dot_key(data: dict, dot_key: str) -> list:
    keys = dot_key.split(".")
    return agg_items_by_keys(data, keys)


def agg_items_by_keys(data: dict, keys: list[str]) -> list:
    items = []
    if len(keys) == 1:
        if isinstance(data[keys[0]], list):
            items.extend(data[keys[0]])
        else:
            items.append(data[keys[0]])
    else:
        for key in keys[:-1]:
            if isinstance(data[key], list):
                for item in data[key]:
                    items.extend(agg_items_by_keys(item, keys[1:]))
            else:
                items.extend(agg_items_by_keys(data[key], keys[1:]))
    return items


def _extra_parser_for_key(keys: list[str], outputs: dict, parser: Callable):
    if len(keys) == 1:
        outputs[keys[0]] = parser(outputs[keys[0]])
    if len(keys) > 1:
        if isinstance(outputs[keys[0]], list):
            for i, item in enumerate(outputs[keys[0]]):
                outputs[keys[0]][i] = _extra_parser_for_key(keys[1:], outputs[keys[0]][i], parser)
        else:
            outputs[keys[0]] = _extra_parser_for_key(keys[1:], outputs[keys[0]], parser)
    return outputs


def extra_parser_for_key(key: str, outputs: dict, parser: Callable) -> dict:
    keys = key.split(".")
    return _extra_parser_for_key(keys, outputs, parser)


def extra_parser_for_keys(keys: list[str], outputs: dict, parser: Callable) -> dict:
    for key in keys:
        outputs = extra_parser_for_key(key, outputs, parser)
    return outputs


def prepare_template_from_instruction(
        instruction: str,
        output_keys: list[str] = None,
        output_dict: dict = None,
        examples: bool = False,
) -> str:
    """Prepare yaml instruction prompt"""

    template = (
        "{instruction}\n\n"
        "=== Response format instructions ===\n\n"
        "Please return a response in yaml format:\n"
        "{output_dict}\n\n"
        "Do not include any other information or explanation to your response, so that your response can be "
        "parsed with yaml.safe_load(). "
        "When values are multiline strings or contain ':' you should set the value in quotes using the Double quotation marks.\n\n"
        "=== End instructions ===\n\n"
        "{remarks}\n\n"
        "Response:"
    )
    if output_dict:
        output_dict = yaml.dump(output_dict)
    elif output_keys:
        output_dict = yaml.dump({key: "<value (string)>" for key in output_keys})
    if not output_keys and not output_dict:
        raise ValueError("Either output_keys or output_dict must be provided")
    if examples:
        remarks = "\n{examples}\n\n{remarks}"
    else:
        remarks = "{remarks}"
    return template.format(
        instruction=instruction,
        output_dict=output_dict,
        remarks=remarks,
    )


TEMPLATE = (
    "{instruction}\n\n"
    "{examples}\n\n"
    "{remarks}\n\n"
    "Begin!\n"
    "{input}\n"
    "---"
)


def prepare_template_from_instruction_and_examples(
        instruction: str,
        output_keys: list[str] = None,
        output_dict: dict = None,
        template: str = TEMPLATE,
) -> str:
    """Prepare yaml instruction prompt"""

    if output_dict:
        output_dict = yaml.dump(output_dict)
    elif output_keys:
        output_dict = yaml.dump({key: "<value (string)>" for key in output_keys})
    if not output_keys and not output_dict:
        raise ValueError("Either output_keys or output_dict must be provided")
    return template.format(
        instruction=instruction,
        examples="{examples}",
        remarks="{remarks}",
        input="{input}",
    )

from __future__ import annotations
import typing as t
import json
from xml.etree import ElementTree as etree
from dataclasses import asdict

import openai

from opendigger_pycli.console import CONSOLE
from opendigger_pycli.config.utils import (
    has_openai_api_key,
    get_openai_api_key_from_config,
)
from opendigger_pycli.datatypes import (
    NonTrivalNetworkInciatorData,
    TrivialNetworkIndicatorData,
    NonTrivialIndicatorData,
    TrivialIndicatorData,
)


def analyze_indicators_data(
    indicator_data: t.List[
        t.Union[
            TrivialIndicatorData,
            NonTrivialIndicatorData,
            TrivialNetworkIndicatorData,
            NonTrivalNetworkInciatorData,
        ]
    ],
) -> t.Dict[str, t.Union[str, t.List[str]]]:
    system_message = r"""
    You are a Github open source data insight expert,
    you will be given some indicators data and their descriptions
    (sometimes the description or data may not be accurate enough,
    you need to further understand the indicators data based on the indicators name), 
    please output a detailed, expert-level insight report 
    for the open source project based on the indicators data 
    (you need to analyze the indicators data in detail), 
    without redundant explanations.
    
    My data format is as follows:
    
    ```xml 
    <all_indicator>
    <indicator>
    <indicator_name>xxxxx</indicator_name> 
    <indicator_description>xxxxx</indicator_description>
    <indicator_data>[{year: int, month: int, value: int | float | list | dict, is_raw: bool}, ...]</indicator_data>
    </indicator>
    <indicator>
    <indicator_name>xxxxx</indicator_name> 
    <indicator_description>xxxxx</indicator_description>
    <indicator_data>[{year: int, month: int, value: int | float | list | dict, is_raw: bool}, ...]</indicator_data>
    </indicator>
    </all_indicator>
    ```
    If you come across data for the same year and month, but is_raw is True, 
    you need to ignore the data where is_raw is False and use the data where is_raw is True. 
    Other than that you don't need to focus on is_raw.
    
    Could you please output the following json format:
    
    ```xml
    <insights>
    <indicator>
    <indicator_name>xxxxx</indicator_name>
    <insight>xxxxx</insight>
    </indicator>
    
    <indicator>
    <indicator_name>xxxxx</indicator_name>
    <insight>xxxxx</insight>
    </indicator>
    
    <summary>xxxxx</summary>
    </insights>
    ``` 
    where <indicator_name> corresponds to the name of the input indicator data,
    <insight> is your output insight report which should analyze the data and give conclusions,
    and <summary> is a summary of the analysis given for all indicators.
    """

    user_message_template = """
    <indicator>
    <indicator_name>{}</indicator_name> 
    <indicator_description>{}</indicator_description>
    <indicator_data>{}</indicator_data>
    </indicator>
    """

    if not openai.api_key:
        if not has_openai_api_key():
            return {"error": "No OpenAI API key found."}
        openai.api_key = get_openai_api_key_from_config()

    unsupported_indicator_names = []
    all_indicator_name = []

    indicator_dat_xmls = []
    for indicator_dat in indicator_data:
        all_indicator_name.append(indicator_dat.name)

        if isinstance(indicator_dat, TrivialNetworkIndicatorData) or isinstance(
            indicator_dat, NonTrivalNetworkInciatorData
        ):
            unsupported_indicator_names.append(indicator_dat.name)
            continue
        if isinstance(indicator_dat, NonTrivialIndicatorData):
            recent_indciator_dat_value = {}
            for key, value in indicator_dat.value.items():
                recent_indciator_dat_value[key] = [asdict(v) for v in value[-12:]]  # type: ignore
            indicator_dat_xmls.append(
                user_message_template.format(
                    indicator_dat.name,
                    indicator_dat.__doc__,
                    json.dumps(
                        recent_indciator_dat_value, separators=(",", ":"), indent=None
                    ),
                )
            )
            continue
        indicator_dat_xmls.append(
            user_message_template.format(
                indicator_dat.name,
                indicator_dat.__doc__,
                json.dumps(
                    [asdict(v) for v in indicator_dat.value[-12:]],
                    separators=(",", ":"),
                    indent=None,
                ),  # type: ignore
            )
        )
    user_message = "<all_indicator>\n"
    for indicator_dat_xml in indicator_dat_xmls:
        user_message += indicator_dat_xml
        user_message += "\n"
    user_message += "</all_indicator>"

    with CONSOLE.status(f"[bold green]Analyzing Indicators: {all_indicator_name}..."):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

    result: t.Dict[str, t.Union[str, t.List[str]]] = {}
    failed_indcators = []
    insights = completion["choices"][0]["message"]["content"]  # type: ignore

    try:
        rv_data = etree.XML(insights)
    except etree.ParseError:
        return {"error": "Failed to parse the response from OpenAI API."}

    indicator_eles = rv_data.findall("indicator")
    for indicator_ele in indicator_eles:
        indicator_name_ele = indicator_ele.find("indicator_name")
        if indicator_name_ele is None or indicator_name_ele.text is None:
            continue
        indicator_insight_elel = indicator_ele.find("insight")
        if indicator_insight_elel is None or indicator_insight_elel.text is None:
            failed_indcators.append(indicator_name_ele.text)
            continue
        result[indicator_name_ele.text] = indicator_insight_elel.text
    summary_ele = rv_data.find("summary")
    if summary_ele is not None and summary_ele.text is not None:
        result["summary"] = summary_ele.text
    else:
        result["summary"] = "No summary found."

    result["unsupported_indicator_names"] = unsupported_indicator_names

    handled_indicators = set(result.keys())
    result["failed_indcators"] = failed_indcators + list(
        set(all_indicator_name) - handled_indicators
    )

    return result

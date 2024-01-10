"""
This module implement the tool for the LLM agent that downloads the charts from the ECMWF Charts API at https://charts.ecmwf.int/opencharts-api/v1. 
"""
import functools
import os
from datetime import datetime
from difflib import SequenceMatcher, ndiff
from typing import List

import requests
from langchain.schema import AIMessage

from ..completion_prompt import create_prompt, get_chatgpt_completion
from ..config import Logger, configs
from ..data_models.chart import Format, Projection, ValidTime
from ..data_models.product import Products, nn, st_model
from ..dbs import dbs

models = {
    "time": ValidTime,
    "product": Products,
    "region": Projection,
    "chat_format": Format,
}


def get_timestamp(user_input: str, params: List[str]) -> str:
    """
    Function that takes the closest datetime from those available in the API implementation.

    Args:
    -----
       user_input: str
       params: List[str]

    Returns:
    --------
       best_match: str
                   the closest datetime from user_input amongst those in params
    """
    user_input = datetime.strptime(user_input, "%Y-%m-%dT%H:%M:%SZ")

    deltas = list()
    min_v = 1e10
    best_match = None
    for param in params:
        param = datetime.strptime(param, "%Y-%m-%dT%H:%M:%SZ")
        delta = user_input - param
        total_seconds = abs(delta.total_seconds())
        if total_seconds < min_v:
            min_v = total_seconds
            best_match = param.strftime("%Y-%m-%dT%H:%M:%SZ")
    return best_match


def strip_non_alphanumerical_characters(s: str) -> str:
    return "".join(filter(str.isalnum, s))


def validate_inputs(func):
    @functools.wraps(func)
    def wrapper_validation(*args, **kwargs):
        server = configs.ECMWF_CHARTS_SERVER
        top_k = nn.kneighbors(
            st_model.encode(kwargs["product"]).reshape(1, -1), return_distance=False
        )[0]
        ask_product_subset = create_prompt(
            kwargs["product"], [list(Products)[product].value[0] for product in top_k]
        )
        Logger.debug(f"Prompt product: {ask_product_subset}")
        response = get_chatgpt_completion(ask_product_subset)
        response = strip_non_alphanumerical_characters(response)
        Logger.debug(f"ChatGPT chosen prodcut: {response}")
        this_product = [
            prod.value[1]
            for prod in Products
            if response == strip_non_alphanumerical_characters(prod.value[0])
        ]  # list(Products)[top_k[0]].value[1]
        kwargs["product"] = this_product[0]

        api_specs = requests.get(
            f"{server}schema/?product={kwargs['product']}&package=openchart"
        )
        specs = None
        if api_specs.status_code == 200:
            specs = api_specs.json()
            specs["paths"].pop("/products")
            key = list(specs["paths"].keys())[0]
            parameters = specs["paths"][key]["get"]["parameters"]
            for param in parameters:
                if param["name"] == "projection":
                    prompt = create_prompt(kwargs["region"], param["schema"]["enum"])
                    response = get_chatgpt_completion(prompt)
                    kwargs["region"] = response
                elif param["name"] == "valid_time":
                    kwargs["time"] = get_timestamp(
                        kwargs["time"], param["schema"]["enum"]
                    )
        output = func(*args, **kwargs)
        return output

    return wrapper_validation


@validate_inputs
def charts(time, product, region, chart_format):
    """
    This tool can be used to retrieve a chart from ECMWF.

    The charts are graphical representations of various atmospheric parameters
    and weather conditions at different levels of the atmosphere.

    These charts are based on numerical weather prediction models, which use
    complex mathematical equations and computer simulations to forecast the
    behavior of the atmosphere.

    Use this tool when asked to download a chart.
    """
    if product is None:
        openapi_ref = dbs[2].similarity_search(query, top_k=1, metric="cos")
        if openapi_ref:
            schema = openapi_ref[0].metadata["schema"]
            r = requests.get(schema)
            if r.status_code == 200:
                info = r.json()
                server = info["servers"][0]["url"]
                product = [key for key in info["paths"].keys() if key != "/products"][0]
                parameters = info["paths"][product]["get"]["parameters"]
            else:
                return None
        Logger.info(f"Charts: {time}, {endpoint}, {projection}")

    server = "https://charts.ecmwf.int/opencharts-api/v1"
    url = "{}/products/{}/?valid_time={}&projection={}".format(
        server, product, time, region.replace("'", "")
    )
    charts_href = None
    Logger.info(f"Url for charts: {url}")
    for i in range(4):
        r = requests.get(url)
        if r.status_code == 200:
            charts_href = r.json()["data"]["link"]["href"]
            Logger.info(f"Found chart href: {charts_href}")
            break
        else:
            if r.json():
                charts_href = r.json()["error"]
            else:
                charts_href = r.text
                return AIMessage(charts_href)

    return charts_href

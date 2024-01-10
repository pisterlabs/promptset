import copy
import json
import logging
import os
import re
import shlex
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Callable

import markdown_strings
import openai
import pandas as pd
import rich.console
import rich.markdown
import tiktoken
from dotenv import load_dotenv
from hubmap_sdk import SearchSdk
from hubmap_sdk.sdk_helper import HTTPException
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from hubmapbot.utils import reduce_tokens_below_limit, kwargs_to_dict, is_running_in_notebook

from .prompts import *

load_dotenv(dotenv_path='.env')

logger = logging.getLogger(__name__)

MODEL_MAP = {
    OpenAIModels.GPT_4: "gpt-4-0613",
    OpenAIModels.GPT_3_5: "gpt-3.5-turbo-0613",
}

DEFAULT_MODEL = OpenAIModels.GPT_4


def chat_completion_default_kwargs(model_name):
    return {
        "model": model_name,
    }


ENCODING = tiktoken.encoding_for_model(DEFAULT_MODEL)

SERVICE_URL = "https://search.api.hubmapconsortium.org/v3/"
SEARCHSDK_INSTANCE = SearchSdk(service_url=SERVICE_URL)

DEBUG_OUT_DIR = Path("debug_out")

data_path = Path("data")
github_path = Path("github")
persist_directory = Path("persist")

vectorstore_info = namedtuple("vectorstore_info", ["name", "path", "col"])

VECTORSTORE_INFOS = {
    "website": vectorstore_info("website", persist_directory / "chroma", "langchain"),
    "es_dataset": vectorstore_info("es_dataset", persist_directory / "chroma_es_dataset", "es_dataset"),
}

MODE_MAP = {
    "Auto 💬": Modes.AUTO,
    "Dataset Search 📊🔎": Modes.SEARCH_DATASET,
    # "Sample Search 🧫🔎": RetrieverOptions.SEARCH_SAMPLE,
    # "Donor Search 👨🔎": RetrieverOptions.SEARCH_DONOR,
    # "Other": RetrieverOptions.OTHER,
    "General Q&A 🤔": Modes.GENERAL,
}





@dataclass
class ChatMessage:
    role: Roles
    summary_content: str
    renderable_content: List[object]
    successful: bool = True
    num_results: int = 0


def hardcoded_explain(parsed_query):
    objs = []

    def traverse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["match", "range", "wildcard"]:
                    objs.append((key, value))
                else:
                    traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)
        else:
            pass

    traverse(parsed_query)

    expl = "Searching for objects that match the following criteria:\n"

    for k, obj in objs:
        if isinstance(obj, dict):
            for field, value in obj.items():
                term = value
                if k == "match":
                    if isinstance(term, dict):
                        term = term.get("query", term.get("value", None))
                    if term is not None and isinstance(term, str):
                        expl += f"- `{field}` matches \"{term}\"\n"
                elif k == "range":
                    if isinstance(term, dict):
                        gte = term.get("gte", None)
                        gt = term.get("gt", None)
                        lte = term.get("lte", None)
                        lt = term.get("lt", None)
                        if gte is not None:
                            expl += f"- `{field}` >= {gte}\n"
                        if gt is not None:
                            expl += f"- `{field}` > {gt}\n"
                        if lte is not None:
                            expl += f"- `{field}` <= {lte}\n"
                        if lt is not None:
                            expl += f"- `{field}` < {lt}\n"
                elif k == "wildcard":
                    if isinstance(term, dict):
                        term = term.get("value", term.get("wildcard", None))
                    if term is not None and isinstance(term, str):
                        escapedterm = markdown_strings.esc_format(term)
                        expl += f"- `{field}` matches \"{escapedterm}\"\n"

    return expl


def _create_retry_decorator(max_retries) -> Callable[[Any], Any]:
    import openai

    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


retry_dec = _create_retry_decorator(max_retries=5)


@retry_dec
def retry_chat_completion(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)


@dataclass
class ParsedQuery:
    query: str


def condense_history(prev_user_message, prev_chatbot_response, user_message: str, model_name) -> str:

    if prev_user_message is None or prev_chatbot_response is None:
        return user_message

    prev_chatbot_response = "[some response here]"

    chat_history = f"User: {prev_user_message}\n\nAssistant: {prev_chatbot_response}"

    formatted_prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=chat_history,
        question=user_message
    )

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")

    choice = completion["choices"][0]
    condensed = choice["message"]["content"]

    logger.debug(f"condensed standalone: {condensed}")

    return condensed


class SelectException(Exception):
    def __init__(self, message):
        super().__init__(message)


def select_engine(user_message, model_name):
    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": RETRIEVER_SELECT_PROMPT.format(question=user_message)},
            {"role": "system", "content": "Do not respond to the user's question."},
        ],
        functions=RETRIEVER_SELECT_FUNCTIONS,
        function_call={"name": "categorize_question"},
        temperature=0,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")

    choice = completion["choices"][0]

    if "function_call" not in choice["message"]:
        content = choice["message"]["content"]
        logger.debug(f"select engine content: {content}")
        usr_message = "I couldn't understand your question. Please try rephrasing your question, or try asking a different question."
        raise SelectException(usr_message)
    else:
        function_call = choice["message"]["function_call"]
        arg_dict = json.loads(function_call["arguments"])

        logger.debug(f"Selected function_call: {function_call}")
        logger.debug(f"Response arg_dict: {arg_dict}")

        category_arg = arg_dict["category"]
        msg = f"category: {category_arg}"

        if "entity_type" in arg_dict:
            msg += f", entity_type: {arg_dict['entity_type']}"
        logger.debug(msg)
        SEARCH_MAPPING = {
            "dataset": Modes.SEARCH_DATASET,
            "sample": Modes.SEARCH_SAMPLE,
            "donor": Modes.SEARCH_DONOR,
        }

        engine_name = category_arg

        if category_arg == "search":
            if "entity_type" not in arg_dict:
                raise SelectException(f"Missing entity_type")
            if arg_dict["entity_type"] not in SEARCH_MAPPING:
                raise SelectException(f"Unknown entity type: {arg_dict['entity_type']}")
            engine_name = SEARCH_MAPPING[arg_dict["entity_type"]]

        if engine_name not in REDUCE_ENGINE_MAP:
            raise SelectException(f"Unknown question category: {engine_name}")

    logger.debug(f"selected engine: {engine_name}")

    return engine_name


def auto_engine(user_message, model_name):
    try:
        res = select_engine(user_message, model_name)
    except SelectException as e:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=str(e), renderable_content=[str(e)])
    engine_name = res
    reduced_engine = REDUCE_ENGINE_MAP[engine_name]
    if reduced_engine == Modes.AUTO:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown question category: {engine_name}", renderable_content=[f"Unknown question category: {engine_name}"])
    return ENGINE_MAPPING[reduced_engine](user_message, model_name)


def simple_engine(user_message: str, model_name) -> str:
    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": SIMPLE_ENGINE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")
    choice = completion["choices"][0]
    content = choice["message"]["content"]

    logger.debug(f"simple engine content: {content}")
    return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])


def general_engine(user_message: str, model_name) -> str:
    docs = RETRIEVERS[Modes.GENERAL].get_relevant_documents(user_message)

    doc_token_limit = 2500
    docs = reduce_tokens_below_limit(doc_token_limit, docs)

    logger.debug(f"docs:")
    for doc in docs:
        logger.debug(doc.page_content)

    formatted_prompt = TEXT_ANSWER_PROMPT.format(
        context="\n\n\n".join([doc.page_content for doc in docs]),
        question=user_message
    )

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        # functions=TEXT_ANSWER_FUNCTIONS,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")
    choice = completion["choices"][0]

    if choice["finish_reason"] != "function_call":
        content = choice["message"]["content"]
        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])
    else:
        content = TEXT_ANSWER_FAIL_MSG
        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=[content])


def es_engine(user_message: str, model_name, entity_type=None, retry=1) -> str:
    entity_formatted_mapping = {
        Modes.SEARCH_DATASET: "Dataset",
        Modes.SEARCH_SAMPLE: "Sample",
        Modes.SEARCH_DONOR: "Donor",
    }

    if entity_type not in entity_formatted_mapping:
        return ChatMessage(role=Roles.ASSISTANT, summary_content=f"Unknown entity type: {entity_type}", renderable_content=[f"Unknown entity type: {entity_type}"])

    entity_type_formatted = entity_formatted_mapping[entity_type]
    extras = f"entity_type {entity_type_formatted}"

    logger.debug(f"searching with user_message: {user_message}")

    docs = RETRIEVERS[entity_type].get_relevant_documents(user_message + " " + extras)

    if len(docs) == 0:
        raise Exception("No docs found")

    doc_token_limit = 4000
    # doc_token_limit = 2000
    docs = reduce_tokens_below_limit(doc_token_limit, docs)

    logger.debug(f"docs:")
    for doc in docs:
        logger.debug(doc.page_content)

    if not os.environ.get("DOCKER_CONTAINER", False):
        log_folder = Path(os.environ.get("LOG_DIR", "logs"))
        log_folder.mkdir(exist_ok=True, parents=True)
        with open(log_folder / "out2.log", "w") as f:
            f.write("\n\n\n".join([doc.page_content for doc in docs]))

    if len(docs) == 0:
        raise Exception("All docs were too long")

    context = "\n\n\n".join([doc.page_content for doc in docs])
    context = context.replace("[n]", "")

    formatted_prompt = ES_SEARCH_PROMPT.format(
        context=context,
        question=user_message,
        entity_type=entity_type_formatted,
    )

    MAX_TOKENS_IN_INPUT = 3600
    formatted_prompt = ENCODING.decode(ENCODING.encode(formatted_prompt)[:MAX_TOKENS_IN_INPUT])

    if not os.environ.get("DOCKER_CONTAINER", False):
        log_folder = Path(os.environ.get("LOG_DIR", "logs"))
        log_folder.mkdir(exist_ok=True, parents=True)
        with open(log_folder / "out3.log", "w") as f:
            f.write(formatted_prompt)

    completion = retry_chat_completion(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        n=3,
        # functions=ES_SEARCH_FUNCTIONS,
        **chat_completion_default_kwargs(model_name)
    )
    logger.debug(f"completion: {completion}")

    results = [parse_es_choice(choice, entity_type_formatted) for choice in completion["choices"]]

    num_succeeded = sum([res.successful for res in results])
    logger.debug(f"num_succeeded: {num_succeeded}")

    if num_succeeded == 0 and retry > 0:
        results += es_engine(user_message, model_name, retry=retry - 1)

    # sort results, succeeded first
    results.sort(key=lambda res: res.num_results)
    results.sort(key=lambda res: res.successful, reverse=True)

    return results


def parse_es_choice(choice, entity_type_formatted):
    did_succeed = False

    if choice["finish_reason"] != "function_call":
        content = choice["message"]["content"]

        res = []

        found_json_str = re.search(r"```json(.*?)```", content, re.DOTALL)
        if not found_json_str:
            found_json_str = re.search(r"```(.*?)```", content, re.DOTALL)

        if found_json_str:
            everything_besides_json = re.sub(r"```json(.*?)```", "", content, flags=re.DOTALL)
            everything_besides_json = everything_besides_json.strip()

            res.append(everything_besides_json)

            json_query = found_json_str.group(1).strip()
            logger.debug(f"json_query: {json_query}")
        else:
            json_query = content.strip()
            logger.debug(f"json_query: {json_query}")

        if json_query is None:
            logger.debug("No parsable json query found")
            return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed)

        try:
            parsed_query = json.loads(json_query)
        except json.decoder.JSONDecodeError as e:
            logger.debug(f"JSONDecodeError: {e}")
            msg = "Sorry, I wasn't able to create a valid JSON query. Please try rephrasing your question, or try asking a different question."
            res.append(e)
            res.append(msg)
            return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed)

        if "query" not in parsed_query:
            parsed_query = {"query": parsed_query}
            logger.debug("No query key found in parsed query, adding one")

        logger.debug(parsed_query)
        res.append(ParsedQuery(query=parsed_query))

        num_results = 0

        try:
            search_results = hubmap_search(parsed_query)
            formatted_search_results = format_search_result_to_dataframe(search_results)  # dataframe
            logger.debug(f"search_results: {search_results}")

            res.append(hardcoded_explain(parsed_query))

            num_results = len(formatted_search_results)
            if num_results > 0:
                formatted_results_msg = f"I ran this query and found {num_results} results:"
                res.append(formatted_results_msg)
                res.append(formatted_search_results)
                did_succeed = True
            else:
                res.append(NO_RESULTS_FOUND_MSG)
        except HTTPException as e:
            logger.debug(f"HuBMAP es search failed: {e}")
            msg = "Sorry, I wasn't able to complete your search. Please try rephrasing your question, or try asking a different question."
            res.append(msg)
            res.append(e)

        return ChatMessage(role=Roles.ASSISTANT, summary_content=content, renderable_content=res, successful=did_succeed, num_results=num_results)

    else:
        function_call = choice["message"]["function_call"]
        arguments = json.loads(function_call["arguments"])
        if function_call.name == "failed":
            res = arguments["reason"]
            return ChatMessage(role=Roles.ASSISTANT, summary_content=res, renderable_content=[res], successful=did_succeed)
        else:
            return ChatMessage(role=Roles.ASSISTANT, summary_content=function_call, renderable_content=[function_call], successful=did_succeed)


ALWAYS_INCLUDES = [
    # "url",
    "uuid",
    "hubmap_id",
    # "donor.mapped_metadata.death_event",
    # "donor.mapped_metadata.mechanism_of_injury",
    # "donor.mapped_metadata.sex",
    # "donor.mapped_metadata.age_value",
    # "donor.mapped_metadata.race",
    # "provider_info",
    # "files.type",
    # "source_samples.created_by_user_email",
    # "donor.mapped_metadata.medical_history"
]

REQUEST_EXTRAS = {
    "_source": {
        "excludes": [
            "ancestors",
            "descendants",
            "immediate_ancestors",
            "immediate_descendants",
            "metadata",
            "donor.metadata"
        ],
    },
    "size": 10000,
}


def hubmap_search(request):
    full_request = copy.deepcopy(request)
    for key, value in REQUEST_EXTRAS.items():
        full_request[key] = value

    full_request["_source"]["includes"] = list(set(ALWAYS_INCLUDES + get_checked_fields(full_request)))
    print(full_request["_source"]["includes"])
    search_result = SEARCHSDK_INSTANCE.search_by_index(full_request, "portal")
    return search_result


def get_checked_fields(request):
    query_body = request["query"]
    # recusive find any objects with key "match", "range", or "wildcard"

    checked_fields = []

    def get_checked_fields_recursive(query_body):
        if isinstance(query_body, dict):
            for key in query_body:
                if key in ["match", "range", "wildcard"]:
                    checked_fields.append(list(query_body[key].keys())[0])
                else:
                    get_checked_fields_recursive(query_body[key])
        elif isinstance(query_body, list):
            for item in query_body:
                get_checked_fields_recursive(item)

    get_checked_fields_recursive(query_body)

    return checked_fields


def get_field_example_dict(json_obj, field_example_dict, prefix=""):
    if isinstance(json_obj, dict):
        for key in json_obj:
            if prefix == "":
                new_prefix = key
            else:
                new_prefix = prefix + "." + key
            get_field_example_dict(json_obj[key], field_example_dict, new_prefix)
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            new_prefix = prefix + f"[{i}]"
            get_field_example_dict(item, field_example_dict, new_prefix)
    else:
        field_example_dict[prefix] = json_obj


def format_search_result_to_dataframe(search_result):
    hits = search_result["hits"]["hits"]

    hits_firstval = []
    for hit in hits:
        val_dict = defaultdict(None)
        get_field_example_dict(hit["_source"], val_dict)
        hits_firstval.append(val_dict)

    df = pd.DataFrame(hits_firstval)

    if "uuid" in df.columns:
        df["url"] = df["uuid"].apply(lambda x: f"https://portal.hubmapconsortium.org/browse/dataset/{x}")

    columns = list(df.columns)
    columns.sort()
    if "url" in columns:
        columns.remove("url")
        columns.insert(0, "url")
    df = df[columns]
    return df


SEARCH_SAMPLE_REQUEST = {
    "query": {
        "bool": {
            "must": [
                {"match": {"provider_info": "Stanford"}},
                {"match": {"entity_type": "Dataset"}}
            ]
        }
    },
    "size": 5
}

PYTHON_SEARCH_SAMPLE_TEMPLATE = """
# !pip install hubmap-sdk
import json
from hubmap_sdk import SearchSdk

service_url = "{service_url}"
searchsdk_instance = SearchSdk(service_url=service_url)

# for more info on elasticsearch query syntax, see https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
search_json = \"\"\"
{search_json}
\"\"\"

search_result = searchsdk_instance.search_by_index(json.loads(search_json), "portal")

print(json.dumps(search_result, indent=2))
"""


def python_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict, indent=2)
    return PYTHON_SEARCH_SAMPLE_TEMPLATE.format(search_json=search_json, service_url=service_url)


CURL_SEARCH_SAMPLE_TEMPLATE = """
curl -X POST "{service_url}search" -H "accept: application/json" -H "Content-Type: application/json" -d {search_json}
"""


def curl_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict)
    return CURL_SEARCH_SAMPLE_TEMPLATE.format(search_json=shlex.quote(search_json), service_url=service_url)


R_SEARCH_SAMPLE_TEMPLATE = """
library(httr)
library(jsonlite)

service_url <- "{service_url}"

search_json <- '{search_json}'

response <- POST(paste0(service_url, "search"), body = search_json, encode = "json", verbose())

content(response, "text")
"""


def r_example_search(search_dict=SEARCH_SAMPLE_REQUEST, service_url=SERVICE_URL):
    search_json = json.dumps(search_dict)
    return R_SEARCH_SAMPLE_TEMPLATE.format(search_json=search_json, service_url=service_url)


EMBEDDINGS = OpenAIEmbeddings(openai_api_key="placeholder")

VECTORSTORES = {}
for vectorstore_name, vectorstore_info in VECTORSTORE_INFOS.items():
    print(f"loading database {vectorstore_name} from {vectorstore_info.path} with collection name {vectorstore_info.col}")

    vectorstore = Chroma(persist_directory=str(vectorstore_info.path), embedding_function=EMBEDDINGS, collection_name=vectorstore_info.col)
    VECTORSTORES[vectorstore_name] = vectorstore

RETRIEVERS = {
    Modes.GENERAL: VECTORSTORES["website"].as_retriever(search_kwargs=kwargs_to_dict(k=20)),
    Modes.SEARCH_DATASET: VECTORSTORES["es_dataset"].as_retriever(search_kwargs=kwargs_to_dict(k=100)),
}

ENGINE_MAPPING = {
    Modes.AUTO: auto_engine,
    Modes.GENERAL: general_engine,
    Modes.SEARCH_DATASET: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET),
    Modes.SEARCH_SAMPLE: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET), # TODO
    Modes.SEARCH_DONOR: lambda *args: es_engine(*args, entity_type=Modes.SEARCH_DATASET), # TODO
    Modes.OTHER: simple_engine,
}

REDUCE_ENGINE_MAP = {
    Modes.AUTO: Modes.AUTO,
    Modes.GENERAL: Modes.GENERAL,
    Modes.SDK: Modes.GENERAL,
    Modes.INGEST: Modes.GENERAL,
    Modes.SEARCH_DATASET: Modes.SEARCH_DATASET,
    Modes.SEARCH_SAMPLE: Modes.SEARCH_DATASET,
    Modes.SEARCH_DONOR: Modes.SEARCH_DATASET,
    Modes.OTHER: Modes.OTHER,
    "about": Modes.GENERAL,
    "about_search": Modes.GENERAL,
}


# def display_chatbot_message(message):
#     display_markdown(f'#### Chatbot:\n{message}')
#
#
# def display_user_message(message):
#     display_markdown(f'#### User:\n{message}')
#
#
# def display_markdown(markdown):
#     is_jupyter_notebook = is_running_in_notebook()
#     if is_jupyter_notebook:
#         from IPython.display import display, Markdown
#         display(Markdown(markdown))
#     else:
#         console = rich.console.Console()
#         md = rich.markdown.Markdown(markdown)
#         console.print(md)
#
#
# def main():
#     # logging.basicConfig(level=logging.DEBUG)
#     logging.basicConfig(filename='simple_chatbot.log', filemode='w', level=logging.DEBUG)
#
#     prev_user_message = None
#     prev_chatbot_response = None
#
#     display_chatbot_message(INITIAL_MESSAGE)
#     while True:
#
#         user_message = input("User: ")
#         if len(user_message) == 0:
#             display_chatbot_message("Goodbye!")
#             break
#         display_user_message(user_message)
#         user_message = condense_history(prev_user_message, prev_chatbot_response, user_message)
#         logger.debug(f"Standalone user question: {user_message}")
#
#         chatbot_response = general_engine(user_message)
#
#         for renderable_content in chatbot_response.renderable_content:
#             display_chatbot_message(renderable_content)
#
#         prev_user_message = user_message
#         prev_chatbot_response = chatbot_response
#
#
# if __name__ == "__main__":
#     main()

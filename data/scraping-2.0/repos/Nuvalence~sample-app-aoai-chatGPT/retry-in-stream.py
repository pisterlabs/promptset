import json
import os
import logging
import random

import requests
import time
import openai
from flask import Flask, Response, request, jsonify, send_from_directory, copy_current_request_context, current_app
from dotenv import load_dotenv
from opencensus.ext.azure.log_exporter import AzureLogHandler

load_dotenv()

app = Flask(__name__, static_folder="static")

# -----------------------------------------------------------------------------
# Logging set up
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(c_handler)

if 'APPLICATIONINSIGHTS_CONNECTION_STRING' in os.environ and 'APPLICATIONINSIGHTS_INSTRUMENTATION_KEY' in os.environ:
    logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING and _INSTRUMENTATION_KEY defined, setting AzureLogHandler")
    connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
    instrumentation_key = os.getenv('APPLICATIONINSIGHTS_INSTRUMENTATION_KEY')
    logger.addHandler(AzureLogHandler(connection_string=connection_string, instrumentation_key=instrumentation_key))
else:
    logger.info(
        "APPLICATIONINSIGHTS_CONNECTION_STRING and _INSTRUMENTATION_KEY defined not defined, AzureLogHandler will not be initialized")


# -----------------------------------------------------------------------------
# Static Files
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file('favicon.ico')


@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("static/assets", path)


# -----------------------------------------------------------------------------
# ACS Integration Settings
# -----------------------------------------------------------------------------
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", 5)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", "true")
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")

# -----------------------------------------------------------------------------
# AOAI Integration Settings
# -----------------------------------------------------------------------------
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE",
                                             "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2023-06-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# -----------------------------------------------------------------------------
# Load balanced endpoints
# -----------------------------------------------------------------------------
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME")

AZURE_OPENAI_KEY_NC = os.environ.get("AZURE_OPENAI_KEY_NC")
AZURE_OPENAI_RESOURCE_NC = os.environ.get("AZURE_OPENAI_RESOURCE_NC")
AZURE_OPENAI_MODEL_NC = os.environ.get("AZURE_OPENAI_MODEL_NC")
AZURE_OPENAI_MODEL_NAME_NC = os.environ.get("AZURE_OPENAI_MODEL_NAME_NC")

AZURE_OPENAI_KEY_US2 = os.environ.get("AZURE_OPENAI_KEY_US2")
AZURE_OPENAI_RESOURCE_US2 = os.environ.get("AZURE_OPENAI_RESOURCE_NC")
AZURE_OPENAI_MODEL_US2 = os.environ.get("AZURE_OPENAI_MODEL_US2")
AZURE_OPENAI_MODEL_NAME_US2 = os.environ.get("AZURE_OPENAI_MODEL_NAME_US2")

# VERY IMPORTANT - THE ORDER IN THE ARRAYS BELOW MUST MATCH!!!!!
openai_keys = [AZURE_OPENAI_KEY, AZURE_OPENAI_KEY_NC, AZURE_OPENAI_KEY_US2]
openai_resources = [AZURE_OPENAI_RESOURCE, AZURE_OPENAI_RESOURCE_NC, AZURE_OPENAI_RESOURCE_US2]
openai_models = [AZURE_OPENAI_MODEL, AZURE_OPENAI_MODEL_NC, AZURE_OPENAI_MODEL_US2]
openai_model_names = [AZURE_OPENAI_MODEL_NAME, AZURE_OPENAI_MODEL_NAME_NC, AZURE_OPENAI_MODEL_NAME_US2]

# -----------------------------------------------------------------------------
# Endpoint randomization stuff
# -----------------------------------------------------------------------------
max_retries = 3


def generate_endpoint(ndx):
    resource = openai_resources[ndx]
    model = openai_models[ndx]
    api_version = AZURE_OPENAI_PREVIEW_API_VERSION
    return f"https://{resource}.openai.azure.com/openai/deployments/{model}/extensions/chat/completions?api-version={api_version}"


def next_index(ndx):
    return (ndx + 1) % max_retries


# -----------------------------------------------------------------------------
# Everything else
# -----------------------------------------------------------------------------
def is_chat_model():
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k',
                                                                                         'gpt-35-turbo-16k']:
        return True
    return False


def should_use_data():
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY:
        return True
    return False


def prepare_body_headers_with_data(request_messages, ndx):
    # request_messages = request.json["messages"]
    open_ai_key = openai_keys[ndx]
    open_ai_resource = openai_resources[ndx]
    open_ai_model = openai_models[ndx]

    body = {
        "messages": request_messages,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentField": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None
                    },
                    "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_SEARCH_TOP_K,
                    "queryType": "semantic" if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" else "simple",
                    "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            }
        ]
    }

    chatgpt_url = f"https://{open_ai_resource}.openai.azure.com/openai/deployments/{open_ai_model}"
    if is_chat_model():
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview"
    else:
        chatgpt_url += "/completions?api-version=2023-03-15-preview"

    headers = {
        'Content-Type': 'application/json',
        'api-key': open_ai_key,
        'chatgpt_url': chatgpt_url,
        'chatgpt_key': open_ai_key,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/1.0.0"
    }

    return body, headers


def stream_with_data(request):
    s = requests.Session()
    response = {
        "id": "",
        "model": "",
        "created": 0,
        "object": "",
        "choices": [{
            "messages": []
        }]
    }

    current_index = random.randint(0, 2)
    start_time = time.time()
    try:
        for retry in range(max_retries):
            try:
                logger.info(f"stream_with_data: starting POST with retries, index = {current_index}")
                endpoint = generate_endpoint(current_index)
                body, headers = prepare_body_headers_with_data(request.json["messages"], current_index)
                r = s.post(endpoint, json=body, headers=headers, stream=True, timeout=5)
                r.raise_for_status()
                break
            except requests.exceptions.Timeout:
                logger.error(f"stream_with_data: POST timed out on attempt {retry}")
                if retry < max_retries - 1:
                    current_index = next_index(current_index)
                    logger.error(f"stream_with_data: retrying, new index = {current_index}")
                else:
                    logger.error("stream_with_data: giving up and returning an error")
                    yield json.dumps({"error": "Sorry, I could not answer that. Please try asking a different question"}) + "\n"
                    return
            except requests.exceptions.RequestException as e:
                logger.error("stream_with_data: an error occurred:", e)
                yield json.dumps({"error": "Sorry, I could not answer that. Please try asking a different question"}) + "\n"
                return

        with r:
            total_time = round(time.time() - start_time, 3)
            logger.info(f"stream_with_data: POST completed in {total_time} seconds, processing response lines")
            start_time = time.time()
            for line in r.iter_lines(chunk_size=10):
                if line:
                    line_json = json.loads(line.lstrip(b'data:').decode('utf-8'))
                    if 'error' in line_json:
                        yield json.dumps(line_json).replace("\n", "\\n") + "\n"
                    response["id"] = line_json["id"]
                    response["model"] = line_json["model"]
                    response["created"] = line_json["created"]
                    response["object"] = line_json["object"]

                    role = line_json["choices"][0]["messages"][0]["delta"].get("role")
                    if role == "tool":
                        response["choices"][0]["messages"].append(line_json["choices"][0]["messages"][0]["delta"])
                    elif role == "assistant":
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                    else:
                        deltaText = line_json["choices"][0]["messages"][0]["delta"]["content"]
                        if deltaText != "[DONE]":
                            response["choices"][0]["messages"][1]["content"] += deltaText

                    yield json.dumps(response).replace("\n", "\\n") + "\n"
            total_time = round(time.time() - start_time, 3)
            logger.info(f"stream_with_data: lines processed in {total_time} seconds")
    except Exception as e:
        logger.error(f"stream_with_data: exception processing response: {str(e)}")
        yield json.dumps({"error": "Sorry, I could not answer that. Please try asking a different question"}) + "\n"


def conversation_with_data(request):
    if not SHOULD_STREAM:
        body, headers = prepare_body_headers_with_data(request.json["messages"], 0)
        endpoint = generate_endpoint(0)
        logger.info(f"Not streaming, using endpoint: {endpoint}")
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        r = r.json()
        return Response(json.dumps(r).replace("\n", "\\n"), status=status_code)
    else:
        logger.info("Streaming, load balancing endpoints")
        if request.method == "POST":
            return Response(stream_with_data(request), mimetype='text/event-stream')
        else:
            return Response(None, mimetype='text/event-stream')


def stream_without_data(response):
    responseText = ""
    for line in response:
        deltaText = line["choices"][0]["delta"].get('content')
        if deltaText and deltaText != "[DONE]":
            responseText += deltaText

        response_obj = {
            "id": line["id"],
            "model": line["model"],
            "created": line["created"],
            "object": line["object"],
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": responseText
                }]
            }]
        }
        yield json.dumps(response_obj).replace("\n", "\\n") + "\n"


def conversation_without_data(request):
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = AZURE_OPENAI_KEY

    request_messages = request.json["messages"]
    messages = [
        {
            "role": "system",
            "content": AZURE_OPENAI_SYSTEM_MESSAGE
        }
    ]

    for message in request_messages:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    response = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=float(AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
        top_p=float(AZURE_OPENAI_TOP_P),
        stop=AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        stream=SHOULD_STREAM
    )

    if not SHOULD_STREAM:
        response_obj = {
            "id": response,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }]
            }]
        }

        return jsonify(response_obj), 200
    else:
        if request.method == "POST":
            return Response(stream_without_data(response), mimetype='text/event-stream')
        else:
            return Response(None, mimetype='text/event-stream')


@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    try:
        use_data = should_use_data()
        logger.info(f"conversation: start of conversation, use data = {use_data}")
        if use_data:
            return conversation_with_data(request)
        else:
            return conversation_without_data(request)
    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Main: application starting")
    app.run()

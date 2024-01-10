import json
import os
import logging
import random

import requests
import time
import openai
from flask import Flask, Response, request, jsonify, send_from_directory
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
    logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING and _INSTRUMENTATION_KEY defined not defined, AzureLogHandler will not be initialized")


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
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
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
total_endpoints = 3


def generate_endpoint(ndx):
    resource = openai_resources[ndx]
    model = openai_models[ndx]
    api_version = AZURE_OPENAI_PREVIEW_API_VERSION
    return f"https://{resource}.openai.azure.com/openai/deployments/{model}/extensions/chat/completions?api-version={api_version}"


def next_index(ndx):
    return (ndx + 1) % total_endpoints


# -----------------------------------------------------------------------------
# Everything else
# -----------------------------------------------------------------------------
def is_chat_model():
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k',
                                                                                         'gpt-35-turbo-16k']:
        return True
    return False


def prepare_body_headers_with_data(r, ndx):
    request_messages = r.json["messages"]
    openai_key = openai_keys[ndx]
    openai_resource = openai_resources[ndx]
    openai_model = openai_models[ndx]

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

    chatgpt_url = f"https://{openai_resource}.openai.azure.com/openai/deployments/{openai_model}"
    if is_chat_model():
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview"
    else:
        chatgpt_url += "/completions?api-version=2023-03-15-preview"

    headers = {
        'Content-Type': 'application/json',
        'api-key': openai_key,
        'chatgpt_url': chatgpt_url,
        'chatgpt_key': openai_key,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/1.0.0"
    }

    return body, headers


def stream_with_data(body, headers, endpoint):
    logger.info(f"stream_with_data: {endpoint}")
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

    logger.info("stream_with_data: starting POST with retries")
    start_time = time.time()
    try:
        with s.post(endpoint, json=body, headers=headers, stream=True, timeout=5) as r:
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


def conversation_with_data(headers_arr, body_arr, endpoint_arr):
    current_index = 0 # random.randint(0, total_endpoints - 1)
    for retry in range(total_endpoints):
        try:
            headers = headers_arr[current_index]
            body = body_arr[current_index]
            endpoint = endpoint_arr[current_index]
            # return geez(headers_arr, body_arr, endpoint_arr)
            return Response(stream_with_data(body, headers, endpoint), mimetype='text/event-stream')

            # r = requests.post(endpoint, headers=headers, json=body)
            # status_code = r.status_code
            # r = r.json()
            # return Response(json.dumps(r).replace("\n", "\\n"), status=status_code)

        except requests.exceptions.Timeout:
            logger.error(f"POST timed out on attempt {retry}")
            if retry < total_endpoints - 1:
                logger.error("Retrying...")
            else:
                logger.error("Giving up and returning an error")
                yield json.dumps({"error": "Sorry, I could not answer that. Please try asking a different question"}) + "\n"
                return
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred:", e)
            yield json.dumps({"error": "Sorry, I could not answer that. Please try asking a different question"}) + "\n"
            return



def blah(request):
    body, headers = prepare_body_headers_with_data(request, 0)
    endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"

    if SHOULD_STREAM:
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        r = r.json()

        return Response(json.dumps(r).replace("\n", "\\n"), status=status_code)
    else:
        if request.method == "POST":
            return Response(stream_with_data(body, headers, endpoint), mimetype='text/event-stream')
        else:
            return Response(None, mimetype='text/event-stream')


def geez(headers_arr, body_arr, endpoint_arr):
    current_index = 0
    headers = headers_arr[current_index]
    body = body_arr[current_index]
    endpoint = endpoint_arr[current_index]
    return Response(stream_with_data(body, headers, endpoint), mimetype='text/event-stream')

def compute_endpoints():
    headers_arr = []
    body_arr = []
    endpoint_arr = []
    for i in range(total_endpoints):
        body, headers = prepare_body_headers_with_data(request, i)
        endpoint = generate_endpoint(i)
        headers_arr.append(headers)
        body_arr.append(body)
        endpoint_arr.append(endpoint)
    return headers_arr, body_arr, endpoint_arr

@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    try:
        logger.info(f"conversation: start of conversation")
        headers_arr, body_arr, endpoint_arr = compute_endpoints()
        return conversation_with_data(headers_arr, body_arr, endpoint_arr)
        # return geez(headers_arr, body_arr, endpoint_arr)
        # return blah(request)
    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Main: application starting")
    app.run()

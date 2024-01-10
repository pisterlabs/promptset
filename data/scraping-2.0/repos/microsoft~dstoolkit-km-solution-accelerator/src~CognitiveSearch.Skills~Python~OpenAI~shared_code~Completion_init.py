# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import azure.functions as func
import json
import os
import openai

# Environment Variables
oai_endpoint = os.environ["OPENAI_ENDPOINT"]
oai_key = os.environ["OPENAI_KEY"]
oai_version = os.environ["OPENAI_VERSION"]
oai_engine = os.environ["OPENAI_ENGINE"]

openai.api_type = "azure"
openai.api_base = oai_endpoint
openai.api_version = oai_version
openai.api_key = oai_key

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info(f'{context.function_name} HTTP trigger function processed a request.')
    if hasattr(context, 'retry_context'):
        logging.info(f'Current retry count: {context.retry_context.retry_count}')
        
        if context.retry_context.retry_count == context.retry_context.max_retry_count:
            logging.info(
                f"Max retries of {context.retry_context.max_retry_count} for "
                f"function {context.function_name} has been reached")

    try:
        body = json.dumps(req.get_json())
    except ValueError:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )
    
    if body:
        result = compose_response(req.headers, body)
        return func.HttpResponse(result, mimetype="application/json")
    else:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )

def compose_response(headers, json_data):
    values = json.loads(json_data)['values']
    
    # Prepare the Output before the loop
    results = {}
    results["values"] = []

    for value in values:
        output_record = transform_value(headers, value)
        if output_record != None:
            results["values"].append(output_record)

    return json.dumps(results, ensure_ascii=False)

## Perform an operation on a record
def transform_value(headers, record):
    try:
        recordId = record['recordId']
    except AssertionError  as error:
        return None

    # Validate the inputs
    try:
        document = {}
        document['recordId'] = recordId

        document['data'] = {}

        assert ('data' in record), "'data' field is required."
        data = record['data']

        # OpenAI tokens
        # Our models understand and process text by breaking it down into tokens. The total amount of content the models can process in a single API call is also limited by the total number of tokens in the call.

        # Get the prompt from record/data
        prompt = data['prompt']

        if 'stop' in data:
            stop = data['stop']
        else:
            stop = None

        if 'temperature' in data:
            temperature = data['temperature']
        else:
            temperature = 0.3

        if 'max_tokens' in data:
            max_tokens = data['max_tokens']
        else:
            max_tokens = 256

        response = openai.Completion.create(
            engine=oai_engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=stop
        )

        # Check status 
        document['data']['response'] = response

    except KeyError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "KeyError:" + error.args[0] }   ]       
            })
    except AssertionError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "AssertionError:" + error.args[0] }   ]       
            })
    except SystemError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "SystemError:" + error.args[0] }   ]       
            })

    return (document)

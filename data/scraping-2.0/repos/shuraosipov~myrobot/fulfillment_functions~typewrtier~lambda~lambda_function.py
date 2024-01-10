import os
import json
import openai


def get_slots(intent_request) -> dict:
    return intent_request["sessionState"]["intent"]["slots"]


def get_slot(intent_request, slotName) -> str:
    slots = get_slots(intent_request)
    if slots is not None and slotName in slots and slots[slotName] is not None:
        return slots[slotName]["value"]["interpretedValue"]
    else:
        return None


def get_session_attributes(intent_request) -> dict:
    sessionState = intent_request["sessionState"]
    if "sessionAttributes" in sessionState:
        return sessionState["sessionAttributes"]

    return {}


def close_action(session_id, intent_name, message):
    """
    This function returns a response back to Lex.
    It is used to close the intent and return a message Lex.
    """

    # This is the response that will be returned to Lex
    # It contains only required fields
    # See https://docs.aws.amazon.com/lexv2/latest/dg/lambda.html?icmpid=docs_console_unmapped#lambda-response-format
    # for more details
    return {
        "sessionState": {
            "dialogAction": {
                "type": "Close",
            },
            "intent": {
                "state": "Fulfilled",
                "name": intent_name,
            },
        },
        "messages": [{"contentType": "PlainText", "content": message}],
        "sessionId": session_id,
    }


def format_message(title, openai_response):
    message = f"""#########\n\n
Title: {title}\n\n
{openai_response}
"""
    return message


def call_openai_api(question, max_tokens=1000, temperature=1) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].text


def lambda_handler(event, context):
    # parsing incoming event and extracting parameters
    intent_request = event
    session_id = intent_request["sessionId"]
    intent_name = intent_request["sessionState"]["intent"]["name"]

    title = get_slot(intent_request, "title")
    #return_message = format_message(title, call_openai_api(title))
    return_message = call_openai_api(title)
    response = close_action(session_id, intent_name, return_message)

    print(json.dumps(response))
    return response

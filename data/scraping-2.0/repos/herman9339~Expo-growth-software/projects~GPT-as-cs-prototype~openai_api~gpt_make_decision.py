import openai
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from openai_api.system_message_and_functions import *
import json
import threading

class NoFunctionInResponseError(Exception):
    """Raised when the GPT response doesn't contain a function call."""
    pass


def call_gpt_default(messages, functions, model="gpt-4"):
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.1,
        max_tokens=500,
        top_p=0.2
    )
    return gpt_response

def process_gpt_response(gpt_response, messages, available_functions):
    response_message = gpt_response["choices"][0]["message"]
    if response_message.get("function_call"):
        handle_function_call(response_message, messages, available_functions)
    else:
        # Raising custom exception if no function call is found in the response message
        raise NoFunctionInResponseError("The response_message did not contain a function call.")
    
    return messages, response_message


def handle_function_call(response_message, messages, available_functions):
    function_name = response_message["function_call"]["name"]
    
    if function_name in available_functions:
        arguments_dict = json.loads(response_message["function_call"]["arguments"])
        function_response = str(available_functions[function_name](arguments_dict))
        logger.info(f"Called function {function_name}")
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
    else:
        logger.warning(f"Unknown function call: {function_name}")




def gpt_make_decision(email, available_functions, system_messages):
    logger.info(f"Processing email: {email['id']}")
    prompt = f"Here is the email {email}. How should we respond?"
    messages = [{"role": "system", "content": system_messages},
                {"role": "user", "content": prompt}]
    
    loop_count = 0
    decision_made = False
    debug_log = {}
    
    while loop_count < 4 and not decision_made:
        try:
            gpt_response = call_gpt_default(messages, functions)
            messages, response_message  = process_gpt_response(gpt_response, messages, available_functions)
        except NoFunctionInResponseError as e:
            logger.error(str(e))
            # You can decide if you want to break out of the loop or continue trying
            break
        
        debug_log[f"loop {loop_count}"] = messages

        if response_message.get("function_call") and response_message["function_call"]["name"] == "decision_made_with_email_info":
            decision_made = True
            logger.info("GPT has made the decision and used the decision_made_with_email_info function.")

        loop_count += 1

    if not decision_made:
        logger.warning("decision was not made")
    
    logger.info("decision is made.")
    return decision_made, gpt_response, debug_log



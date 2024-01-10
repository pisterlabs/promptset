import json
import time
import re
import openai

from core import plugin_registry
from core import speech_utils

with open('openai_API_key.txt', 'r') as file:
    openai.api_key = file.readline().strip()

def sentence_stream_from_token_stream(token_stream):
    # hack to remove the JSON mess that comes before the first sentence
    def remove_json_start(sentence):
        json_start = "\"message\": \""
        if json_start in sentence:
            return sentence[sentence.index(json_start)+len(json_start):]
        return sentence

    def remove_json_end(sentence):
        # more hacks to remove JSON at the end. This is a little tricker than above because
        # the formatting is inconsistent and we need to match the whole terminating string, not just the "message" part
        json_end = r'["\']\s*\}'
        match = re.search(json_end, sentence)
        if match:
            return sentence[:match.start()]
        return sentence
    
    def find_sentence_end(buffer):
        # Identify potential sentence-ending positions
        positions = [buffer.find(p) for p in ['. ', '! ', '? ']]
        
        # Filter out positions that are either -1 or have a space right before them, plus those with "Mr" or "Ms" right before them
        valid_positions = [pos for pos in positions if pos > 0 and buffer[pos-1] != ' ' and buffer[pos-2:pos] != "Mr" and buffer[pos-2:pos] != "Ms"]
        
        return min(valid_positions) if valid_positions else -1

    buffer = ''
    chunk = next(token_stream)
    while chunk["choices"][0]["delta"].get("function_call") is not None:
        buffer += chunk["choices"][0]["delta"]["function_call"]["arguments"]
        first_end = find_sentence_end(buffer)
        while first_end != -1:
            sentence = remove_json_start(buffer[:first_end+1].strip())  # +1 to include the punctuation
            yield sentence
            # Remove the processed part from the buffer
            buffer = buffer[first_end+2:].strip()  # +2 to move past the punctuation and space
            # Find the next valid pattern match in the remaining buffer
            first_end = find_sentence_end(buffer)

        chunk = next(token_stream)

    # If there's anything left in the buffer after all tokens are processed, yield it too
    if buffer:
        buffer = remove_json_end(buffer)
        # if the response is only one sentence, also need to remove the messy start
        buffer = remove_json_start(buffer)
        yield buffer


def speak_streamed_response(interaction_history, token_stream):
    sentence_stream = sentence_stream_from_token_stream(token_stream)
    sentences = []
    for sentence in sentence_stream:
        if not sentence:
            input()
        speech_utils.speak_response(sentence)
        sentences.append(sentence)

    interaction_history[-1]["function_call"]["arguments"] = json.dumps({"message": " ".join(sentences)})


def handle_normal_function_call(interaction_history, token_stream):
    #handle regular function call
    chunk = next(token_stream)
    while chunk["choices"][0]["delta"].get("function_call") is not None:
        interaction_history[-1]["function_call"]["arguments"] += chunk["choices"][0]["delta"]["function_call"]["arguments"]
        chunk = next(token_stream)

    function_to_call = plugin_registry.plugin_function_registry[interaction_history[-1]["function_call"]["name"]]
    function_output = function_to_call(**json.loads(interaction_history[-1]["function_call"]["arguments"]))
    interaction_history.append({
        "role": "function",
        "name": interaction_history[-1]["function_call"]["name"],
        "content": function_output,
    })

def get_function_call(interaction_history):
    model = "gpt-4"
    got_fn_call = False
    while not got_fn_call:
        token_stream = openai.ChatCompletion.create(
            model=model,
            messages=interaction_history,
            functions=plugin_registry.plugin_function_docs,
            function_call="auto",
            temperature=0,
            stream=True
        )
        message = ""
        chunk = next(token_stream)
        # sometimes non-null responses contain empty content. So explicitly check for None
        while chunk["choices"][0]["delta"].get("content") is not None:
            message += chunk["choices"][0]["delta"]["content"] 
            print(chunk["choices"][0]["delta"]["content"] , end="", flush=True)
            chunk = next(token_stream)

        # TODO: handle this more elegantly
        if chunk["choices"][0].get("finish_reason"):
            print("ERROR: DID NOT CALL A FUNCTION")
            interaction_history.append({
                "role": "assistant",
                "content": message if message else None
            })
            interaction_history.append({
                "role": "user",
                "content": "You did not call a function. Please try again."
            })
        else:
            got_fn_call = True
    return message, chunk, token_stream


def handle_single_interaction(interaction_history):
    message, chunk, token_stream = get_function_call(interaction_history)
    # at this point we should be looking at the first chunk of the function call,
    # which always contains the full name of the function being called
    function_name = chunk["choices"][0]["delta"]["function_call"]["name"]
    interaction_history.append({
        "role": "assistant",
        "content": message if message else None,
        "function_call": {
            "name": function_name,
            "arguments": ""
        }
    })
    if function_name == "respond":
        speak_streamed_response(interaction_history, token_stream)
    elif function_name in plugin_registry.plugin_function_registry:
        handle_normal_function_call(interaction_history, token_stream)
    else:
        #TODO: handle this better
        print("ERROR: invalid function call")

    return function_name == "respond"

def work_autonomously_until_done(user_text, interaction_history):
    interaction_history.append({"role": "user", "content": user_text})
    responded_to_user = False
    while not responded_to_user:
        responded_to_user = handle_single_interaction(interaction_history)




    # while response["choices"][0]["message"].get("function_call"):

    #     function_name = response["choices"][0]["message"]["function_call"]["name"]
    #     if function_name not in plugin_registry.plugin_function_registry:
    #         break
    #     function_to_call = plugin_registry.plugin_function_registry[function_name]
    #     function_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
    #     function_response = function_to_call(**function_args)

    #     interaction_history.append(response["choices"][0]["message"])
    #     interaction_history.append({
    #         "role": "function",
    #         "name": function_name,
    #         "content": function_response,
    #     })
    #     response = openai.ChatCompletion.create(
    #         model=model,
    #         messages=interaction_history,
    #         functions=plugin_registry.plugin_function_docs,
    #         function_call="auto",
    #         temperature=0
    #     )
    #     # print("MESSAGE:")
    #     # print(second_response["choices"][0]["message"])
    #     # interaction_history.append(second_response["choices"][0]["message"])
    #     # return second_response["choices"][0]["message"]["content"]

    # interaction_history.append(response["choices"][0]["message"])
    # return response["choices"][0]["message"]["content"]
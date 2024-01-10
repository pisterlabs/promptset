import time
import openai
import pandas as pd
from enum import Enum


openai.api_key = "YOUR_OPENAI_API_KEY"

class Model(Enum):
    CODE = "YOUR_MODEL_TO_GENEREATE_CODE"
    SUMMARY = "YOUR_MODEL_TO_GENEREATE_SUMMARY"
    CHAT = "YOUR_MODEL_TO_GENEREATE_SUMMARY"



def update_message_buffer(request, user_message, response_message):
    user_summary, bot_summary = __get_summary(user_message, response_message)
    messages = request.session.get("messages", [])
    messages.append((user_summary, bot_summary))
    request.session["messages"] = messages

def empty_message_buffer(request):
    length = len(request.session["messages"])
    request.session["messages"] = []


def get_chatbot_response(request, csv_file, user_message, model_type=Model.CHAT):
    assert model_type == Model.CHAT, "Should be a chat model"

    if csv_file is None:
        response = openai.ChatCompletion.create(
        model=model_type.value,
        messages=[
            {"role": "user", "content": f"{user_message}"},
            ]
        )
        return response['choices'][0]['message']['content']
    
    observation_message = __evaluate_data(csv_file)
    messages = request.session.get("messages", [])

    
    
    def __combine_history(messages):
        for message in messages:
            final_message += 'user asked: ' + message[0]
            final_message += 'you answerd: ' + message[1]
        return final_message
    
    final_message = __combine_history(messages, user_message, observation_message)

    response = openai.ChatCompletion.create(
        model=model_type.value,
        messages=[
            {"role": "user", "content": f"{final_message}"},
            ]
    )
    ans = response['choices'][0]['message']['content']
    print(ans)
    return ans

def __evaluate_data(csv_file):
    '''
        This function generate metadata and stat info from the data
    '''
    return 0

def __get_summary(user_message, response_message, model_type=Model.SUMMARY):
    '''
        This function generate summary for bot response and user message
    '''
    assert model_type == Model.SUMMARY, "Should be a summary model"
    return (user_summary, bot_summary)
import openaihandler
import requests
import json
from datetime import datetime

# This file contains the message handler for the bot
def handle_message(contents, sender, jump_url, previous_message_content=None):
    
    response = None;
    if previous_message_content is not None:
        response = openaihandler.get_response(contents, previous_message_content)
    else:
        response = openaihandler.get_response(contents)

    print (response)
    tags = parse_ai_json_tags(str(response))
    ai_comment = parse_ai_json_ai_comment(str(response))
    if tags == ["Safe"]:
        return "This message was safe."
    
    if previous_message_content is not None:
        post_to_database(sender, contents, ai_comment, jump_url, tags, previous_message_content)
    else:
        post_to_database(sender, contents, ai_comment, jump_url, tags)
    return response

def parse_ai_json_tags(response):

    """
    Response is a jsonstring
    AI JSON Format:

    {"tags":["tag1", "tag2", "tag3"], "AI Comment":"This is an AI comment."}
    """

    tags = json.loads(response)["tags"]

    print(tags)

    return tags

    

def parse_ai_json_ai_comment(response):

    """
    Response is a jsonstring
    AI JSON Format:

    {"tags":["tag1", "tag2", "tag3"], "AI Comment":"This is an AI comment."}
    """


    ai_comment = json.loads(response)["AI Comment"]
    print(ai_comment)

    return ai_comment




def get_current_date():
    current_date = datetime.now().strftime('%Y-%m-%d')
    return current_date

def post_to_database(user, message, ai_comment, jump_url, tags, context=""):
    print(type(tags))
    date = get_current_date()

    url = "http://174.3.244.48:8000/api/cases"

    payload = json.dumps({
    "date": str(date),
    "status": "Unreviewed",
    "platform": "Discord",
    "offender": str(user),
    "message": str(message),
    "context": str(context),
    "message_url": str(jump_url),
    "ai_comment": str(ai_comment),
    "ai_tags": tags
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    pass




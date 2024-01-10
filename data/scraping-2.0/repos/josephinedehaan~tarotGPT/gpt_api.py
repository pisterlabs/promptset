import requests
import os
import json
from flask import session
from dotenv import load_dotenv

def fetch_tarot_reading(selected_cards):
    load_dotenv()
    api_key = os.getenv('OPENAI_KEY')

    print(api_key)
    url = 'https://api.openai.com/v1/engines/text-davinci-003/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    prompt = f"Generated cards: {selected_cards} Reply with a ~200 character paragraph: go over the meaning of each card provided and emphasise what its position is (eg past, present, future, etc). Offer some basic insight into how this spread can answer the question that was asked earlier (if a question was asked).  \Do not refer to cards spatially but in relation to the position field provided in JSON. Always refere to the user as 'you' and the tarot reader as 'I'."
    
    data = {
        'prompt': prompt,
        'max_tokens': 350
    }

    if "log" not in session:
        session["log"] = {}
        session["log"]["reading"] = []
     
    session["log"]["reading"].append(prompt)   
    session.modified = True
   
    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if 'choices' in response_data and response_data['choices']:
            message = response_data['choices'][0].get('text', '').strip()
            session["log"]["reading"].append(f"TarotGPT: {message}")   
            session.modified = True
            return message
        else:
            return "Error: Invalid response from OpenAI API" + json.dumps(response_data)
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return None


def remove_tarot_gpt_prefix(input_string):
    prefix = "TarotGPT: "
    if input_string.startswith(prefix):
        return input_string[len(prefix):]
    else:
        return input_string


def chat(message_type, message, card_name):

    if card_name == None:
        card_name = ""

    chat_system_prompts = {
    "reading" : f"You are TarotGPT, a tarot reader. If the user hasn't already asked a question, enquire whether the user would like to ask the tarot any specific questions. \
                    If the user has no more questions, invite the user to press the shuffle cards button. This will provide you a tarot card spread. Once you have received the spread, keep it in memory as no other tarot spread can be generated.\
                    User messages will always end in with the following symbol: '🜑'. \
                    Never end your own messages with this symbol ('🜑'). \
                    ONLY reply as TarotGPT, but never start the reply with the text: \"TarotGPT\".",
                    
    "card_detail": f"You are a tarot card expert. You know alot about tarot cards, but you are not a tarot card. \
                        Answer the user's questions about the meaning of a specific card. " + card_name + " is the card that the user is asking about.\
                        When possible, casually include an emoji that is relevant to the tarot card without mentioning the word Emoji."
     
}

    system_message = chat_system_prompts.get(message_type, None)

    counter = session.get('counter', 0)
    counter += 1
    session['counter'] = counter

    if "log" not in session:
        session["log"] = {}

    log_type = message_type 
    if log_type == "card_detail":
        log_type = card_name


    if log_type not in session["log"]:
        session["log"][log_type] = []
        session["log"][log_type].append(system_message)

    session["log"][log_type].append(f"User: {message}")   

    load_dotenv()
    api_key = os.getenv('OPENAI_KEY')

    print(api_key)
    url = 'https://api.openai.com/v1/engines/text-davinci-003/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    data = {
        'prompt': ' '.join(session["log"][log_type]),
        'max_tokens': 1000
    }

    print(data)

    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if 'choices' in response_data and response_data['choices']:
            message = response_data['choices'][0].get('text', '').strip()
            message = remove_tarot_gpt_prefix(message)
            session["log"][log_type].append(f"TarotGPT: {message}")   

            return f"{message}"
        else:
            return "Error: Invalid response from OpenAI API" + json.dumps(response_data)
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return None





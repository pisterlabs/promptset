from openai_api import run_dalle, get_openai_response, getChatCompletionResponseCommand
from whatsapp_sender import send_whatsapp_image_reply, send_whatsapp_text_reply
from short_term_memory import write_short_term_memory, append_history
import requests
from bs4 import BeautifulSoup
import os
import re
from system_messages import scrape_error_message, get_web_search_safety_prompt, unsafe_google_search_message

def handle_command(command, phone_number_id, from_, history, user_secret, is_private_on, is_unsafe_on):
    if is_unsafe_on:
        send_whatsapp_text_reply(phone_number_id, from_, "Sorry commands are disabled in unsafe mode.", is_private_on, is_unsafe_on)
        history = append_history(history, "system", "There was an error executing the command")
        write_short_term_memory(from_, history, user_secret, is_private_on)
        return

    if command['command_name'] == 'dalle':
        image_prompt = command['image_prompt']
        url = run_dalle(image_prompt)
        send_whatsapp_image_reply(phone_number_id, from_, url)
        history = append_history(history, "system", "The user was given the generated image")
        write_short_term_memory(from_, history, user_secret, is_private_on)
        return

    if command['command_name'] == 'web_search':
        search_prompt = command['search_prompt']
        if not is_google_search_safe(search_prompt):
            send_whatsapp_text_reply(phone_number_id, from_, unsafe_google_search_message(), is_private_on, is_unsafe_on)
            history = append_history(history, "system", "The web search was not performed as it was not safe")
            write_short_term_memory(from_, history, user_secret, is_private_on)
            return
    
        search_result = google_search(search_prompt)
        if search_result is None:
            # Some error happened
            send_whatsapp_text_reply(phone_number_id, from_, scrape_error_message(), is_private_on, is_unsafe_on)
            history = append_history(history, "system", "There was an error doing a web search")
            write_short_term_memory(from_, history, user_secret, is_private_on)
            return
        
        # Append search results and generate a new response
        history = append_history(history, "system", "The web search resulted in the following search results: " + search_result)
        ai_response, _  = get_openai_response(None, history, False)
        send_whatsapp_text_reply(phone_number_id, from_, ai_response, is_private_on, is_unsafe_on)
        history = append_history(history, "assistant", ai_response)
        write_short_term_memory(from_, history, user_secret, is_private_on)
        return


def google_search(term):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ.get("google_search_key"),
        "cx": os.environ.get("google_cx"),
        "q": term
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            r = ""
            for item in response.json()['items'][:3]:
                r = r + scrape_link(item['link'], 250) + "\n\n"
            return r
        except Exception as _:
            return None
    else:
        return None
    
def is_google_search_safe(query):
    messages = [
        {"role": "system", "content": get_web_search_safety_prompt(query)},
    ]
    ai_response, _  = getChatCompletionResponseCommand(messages)
    return ai_response.lower().find("no") == -1

def scrape_link(url, limit):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        all_text = soup.get_text().lower()
        all_text = all_text.replace("\n", " ")
        #all_text = re.sub('[^a-z0-9 °]+', '', all_text)
        words = all_text.split(" ")
        good_words = [string for string in words if len(string) > 2 and len(string) < 20]
        return " ".join(good_words[:limit])
    except Exception as _:
        return ""
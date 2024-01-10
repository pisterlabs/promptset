from openai import RateLimitError
from openai import OpenAI
from decouple import config
import json
import logging
import traceback
import random
logging.basicConfig(filename='gpt.log', level=logging.INFO)


def analyze_text_with_openai(text, filename):
    key_preface = "openai_"
    keys = []
    for i in range(1, 10):
        keys.append(config.get(key_preface + i))
    
    current_key = random.choice(keys)
    client = OpenAI(api_key=current_key)
    content = text

    sys_prompt = "You are a system who analyze dark net data. Your job is to analyze the text, find product name, location, price and domain name from them."
    sys_prompt += "You don't return any explanation, User knows what they are doing. You return a dictionary in this format:"
    sys_prompt += "{product: <product information>, location:<location_info>, price: <price_info> and domain:<domain_info>}"
    sys_prompt += "if you find specific infromation for any category, you put null in that category"
    sys_prompt += "if the domain is in onion or dark site, omit them."
    sys_prompt += "property name in dictionary will be in quotation"
    sys_prompt += "if there are multiple location or cities, you will specify COUNTRY name only"
    sys_prompt += "if there are muiltple price, you will send the average like 10 USD or 1 BTC"

    try:
        print("GPT Analysis: ", filename)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            # model="gpt-3.5-turbo",
            # model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": content
                }],
        )

        message_content = response.choices[0].message.content
        parsed_content = parse(message_content)
        return parsed_content

    except RateLimitError as e:
        error_traceback = traceback.format_exc()
        print("RateLimitError OpenAI:", error_traceback)
        logging.error(f"filename: {filename} - RateLimitError in OpenAI: {e}")
        return {
            "product": None,
            "location": None,
            "price": None,
            "domain": None
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        print("Error while querying OpenAI:", error_traceback)
        logging.error(f"filename: {filename} -Exception occurred in OpenAI: {e} - ")
        return {
            "product": None,
            "location": None,
            "price": None,
            "domain": None
        }

def parse(json_string):
    if '```json' in json_string:
        new_json = json_string[7:]
        new_json = new_json[0: len(new_json) - 3]
        if '\n' in new_json:
            new_json = new_json.replace('\n', '')

        if new_json[0] == 'n':
            new_json = new_json[1:]
        json_string = new_json

    try:
        json_data = json.loads(json_string)
        return json_data
    except Exception:
        error_traceback = traceback.format_exc()
        print("Error while Parsing JSON frm OpenAI:", error_traceback)
        logging.error("Error while Parsing JSON frm OpenAI:", error_traceback)

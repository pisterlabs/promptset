from dotenv import load_dotenv
from ws_data import sort_instagram,sort_pinterest,order_hist
from auto_encoder import cvae_2
import openai
import os
import requests
import json
import re

url = "https://api.openai.com/v1/completions"
load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key

def remove_formatting(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\[\],]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main(input_kw):
    json_filename = 'backend/data/mock_test_data.json'

    with open(json_filename, 'r') as json_file:
        data_test = json.load(json_file)

    insta_data = sort_instagram.main()
    pin_data = sort_pinterest.main()
    user_data = order_hist.main()
    social_trends = insta_data + pin_data
    
    user_data_list = [item.strip() for item in user_data.split(',')]
    input_kw_list = [item.strip() for item in input_kw.split(',')]
    social_trends_list = [item.strip() for item in social_trends.split(',')]

    data_test[0]["social_trends"] = social_trends_list
    data_test[0]["user_data"] = user_data_list
    data_test[0]["Input fashion outfits"]=input_kw_list

    with open(json_filename, 'w') as json_file:
        json.dump(data_test, json_file, indent=2)
            
    ae_kw = cvae_2.main(json_filename)
    combined_prompt = f"{insta_data}{pin_data}{user_data}{input_kw}{ae_kw}"
    formatted_prompt = remove_formatting(combined_prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "text-curie-001",
        "prompt": f"Create a good fashion outfit from these {formatted_prompt} and list out its items.",
        "max_tokens": 80
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    generated_text = response_json["choices"][0]["text"]
    output_lines = generated_text.split('\n')
    output_lines = [line.strip() for line in output_lines if line.strip() != '']

    item_names = []
    for line in output_lines:
        words_in_sentence = [word.strip() for word in line.split(',')]
        if len(words_in_sentence) > 1:
            item_names.extend(words_in_sentence)
        else:
            item_names.append(re.sub(r'^\d+\.\s*', '', line))
    return item_names
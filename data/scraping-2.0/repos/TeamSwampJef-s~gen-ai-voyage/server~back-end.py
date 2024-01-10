from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import openai
import csv
import os
import re

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


client = Elasticsearch(
  'https://verbot.es.us-central1.gcp.cloud.es.io',
    api_key='ENTER API KEY HERE',
)

openai.api_key = 'ENTER API KEY HERE'

plans = '''
Plan Name,Type,Data Limit,Speed,Monthly Price,Description
Verizon Start Unlimited,Mobile,Unlimited,4G LTE, $70,Unlimited talk, text, and data for your smartphone with DVD-quality streaming (480p).
Verizon Play More Unlimited,Mobile,Unlimited,4G LTE, $80,Unlimited talk, text, and data with premium 5G Ultra Wideband and 720p HD streaming.
Verizon Do More Unlimited,Mobile,Unlimited,4G LTE, $80,Unlimited talk, text, and data with 50GB of premium 4G LTE data and 720p HD streaming.
Verizon Get More Unlimited,Mobile,Unlimited,4G LTE, $90,Unlimited talk, text, and data with 50GB of premium 4G LTE data, 720p HD streaming, and Apple Music included.
Verizon 5G Home Internet,Internet,Unlimited,Up to 1 Gbps, $50,High-speed 5G Home Internet with unlimited data and up to 1 Gbps download speeds.
Verizon Fios Internet 200/200,Internet,200 GB,200 Mbps, $39.99,Fios Internet with 200GB data limit and download speeds of 200 Mbps.
Verizon Fios Gigabit Connection,Internet,Unlimited,Up to 940/880 Mbps, $79.99,Fios Gigabit Internet with unlimited data and download speeds of up to 940 Mbps.
'''

@app.route('/api/recommendation', methods=['POST'])
@cross_origin()
def get_response():
    user_message = request.json.get('prompt')

    # TODO: Before you start, make sure you have the following files in the folder embedding\Goal\Devices_csv:
    data = read_csv_files(r"/home/ufasih/workspace/verizon-bot/server/Devices_csv")

    prompt = f"Use the following tables in csv format:\n{format_data_for_prompt(data)}{plans}\n\nPretend you are an AI bot for Verizon, try your best to respond positively with results from the above dataset based on the input prompt from the user. Always try to recommend plans and products that the user is looking for and is most relatable.\n\nUser Prompt:{user_message}\n\nReturn information in a single paragraph, limit results to the top most 1 or 2 only. Use shorter sentences. Do not include too much technical jargon. Aim at the average user. REPLY MUST BE LESS THAN 300 CHARACTERS"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )
    gpt_response = response.choices[0].text.strip()

    textFromChatGPT = gpt_response

    # initialize blocks
    blocks = []

    print(gpt_response)

    def lowercase_and_clean_string(input_string):
        # Convert the string to lowercase
        lowercased_string = input_string.lower()

        # Remove all characters except letters, numbers, spaces, and the dollar sign
        cleaned_string = re.sub(r'[^a-z0-9 $]', '', lowercased_string)

        return cleaned_string

    clean_res = lowercase_and_clean_string(gpt_response)
    try:
        es_response = client.search(
            index="products,plans",
            body={
                "query": {
                    "query_string": {
                        "query": clean_res,
                        "analyzer": "custom_cross_fields_analyzer"
                    }
                },
                "size": 10,
               "_source": ["Name", "Brand", "Price", "ImageURL", "URL", "Category", "DataLimit", "Speed", "Description"]
            }
        )

        blocks +=  [hit.get('_source') for hit in es_response.get('hits', {}).get('hits', [])]

    except es_exceptions.RequestError as e:
        print(f"Encountered an Elasticsearch error: {e}")

    return jsonify({"response": gpt_response, "blocks": blocks})


def read_csv_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            category = filename.replace('.csv', '').replace('Devices_', '')  # 提取类别名
            with open(os.path.join(folder_path, filename), newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                data[category] = list(reader)
    return data


def format_data_for_prompt(data):
    formatted = ""
    for category, items in data.items():
        formatted += f"{category}:\n"
        formatted += "name,price\n"
        for item in items:
            formatted += f"{item['name']},{item['price']}\n"
        formatted += "\n"
    return formatted


if __name__ == '__main__':
    app.run(debug=True)

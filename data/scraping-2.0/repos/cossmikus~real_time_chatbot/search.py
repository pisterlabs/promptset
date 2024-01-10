

import os
import openai
import requests
from pprint import pprint
import textwrap
from flask import jsonify, Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

openai_api_key = ''  # Add your OpenAI API key
bing_search_api_key = ''  # Add your Bing Search API key
bing_search_endpoint = ''


def search(query):
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt}
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    try:
        response = requests.get(bing_search_endpoint, headers=headers, params=params)
        response.raise_for_status()
        json = response.json()
        return json["webPages"]["value"]
    except Exception as ex:
        raise ex


@app.route('/api/home', methods=['GET', 'POST'])
def return_home():
    if request.method == 'POST':
        data = request.get_json()
        word = data.get('word', '')

        results = search(word)

        results_prompts = [
            f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
        ]

        prompt = "Use these sources to answer the question:\n\n" + \
                 "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

        if results:
            openai.api_key = openai_api_key

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500,  # Set the desired maximum tokens
                temperature=1.0,
                n=1,
                stop=None
            )

            response = response["choices"][0]["text"]
            return jsonify({
                'message': response
            })
        else:
            return jsonify({
                'message': "No results found for the given query."
            })
    else:
        return jsonify({
            'message': 'Henry Moragan'
        })


if __name__ == '__main__':
    app.run(debug=True, port=8080)

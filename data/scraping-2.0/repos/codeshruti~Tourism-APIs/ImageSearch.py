from flask import Flask, request, jsonify
import openai
from serpapi import GoogleSearch

# Set up your OpenAI API key
openai.api_key = "########"

app = Flask(__name__)

# Define a function to search images using GoogleSearch API
def search_images(query):
    params = {
        "engine": "google",
        "tbm": "isch",
        "q": query,
        "api_key": "########"
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    if data.get('search_metadata').get('status') == 'Success':
        results = data.get('images_results')
        if results:
            images = []
            for result in results:
                images.append(result['original'])
            return images[:10]
        else:
            return "No results found."
    else:
        return "Search failed. Please try again later."

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    response = search_images(query)
    return jsonify({'images': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

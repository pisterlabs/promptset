import json
import openai
import requests
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__)

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def createReturnMessage(message, command=None):
    return jsonify({"data": {"command": command, "message": message}})

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.json
    preferences = data['preferences']
    movie_or_tvshows = data['movie_or_tvshows']

    text = (
        f"Recommend top 10 recent {movie_or_tvshows} based on {preferences}")
    prompt = (
        f"You will be provided with text delimited by triple backticks, return a JSON with the fields results: {{movie_name:, year:, media_type:{movie_or_tvshows}}} as the response ```{text}```")

    response = get_completion(prompt)
    response = json.loads(response)
    urlPath = "http://tmdb_api_service:5051/get_details"
    try:
        return (requests.post(urlPath, json=response).json())
    except:
        return createReturnMessage("Error: TMDB Server Down for the moment.", command="fetch_movie_details")

if __name__ == '__main__':
    app.run(debug=True, port=5052, host="0.0.0.0")

import json
import re

import openai
from dotenv import dotenv_values
from fetch_movies_db import fetch_movie_from_db_by_imdb_id
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from moviebot import insert_movie, moviebot_chat, search_movies

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# /api/home
@app.route("/api/home", methods=["GET"])
def return_home():
    return jsonify(
        {
            "message": "Hello World!",
        }
    )


@app.route("/api/movies", methods=["GET"])
def get_movies():
    movie_imdb_ids = request.args.getlist("imdb_ids")
    print("Received IMDb IDs:", movie_imdb_ids)
    movies = [fetch_movie_from_db_by_imdb_id(imdb_id) for imdb_id in movie_imdb_ids]
    return jsonify(movies)


@app.route("/api/moviebot", methods=["POST"])
def moviebot():
    try:
        user_msg = request.json.get("message")
        response = moviebot_chat(user_msg)

        print(f"Chatbot Response: {response['bot_msg']}")

        movie_imdb_ids = [movie["id"] for movie in response["recommended_movies"]]

        return jsonify(
            {"bot_msg": response["bot_msg"], "recommended_movies": movie_imdb_ids}
        )

    except Exception as e:
        print("Error in /api/moviebot endpoint:", e)
        return jsonify({"error": "An error occurred while processing the request."})


@app.route("/api/insert_movie", methods=["POST"])
def insert_movie_endpoint():
    movie_data = request.json
    insert_movie(movie_data)
    return jsonify({"message": "Movie inserted successfully!"})


if __name__ == "__main__":
    app.run(debug=True, port=8080)

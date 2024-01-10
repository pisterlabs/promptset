import re

import chromadb
import openai
from dotenv import dotenv_values
from embeddings import get_embedding

# Load environment variables
config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

# Setup Chroma client for local access
chroma_client = chromadb.PersistentClient(path="./chroma")
collection_name = "film_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name)


def insert_movie(movie_data):
    # Generate an embedding for the movie description
    film_embedding = get_embedding(
        movie_data["Description"], model="text-embedding-ada-002"
    )
    # Prepare the document to be inserted
    document = {"document": movie_data, "embedding": film_embedding.tolist()}
    # Insert the document into the collection
    collection.insert(document)


def search_movies(query_description, n=3):
    query_embedding = get_embedding(query_description, model="text-embedding-ada-002")
    results = collection.query(query_embeddings=[query_embedding], n_results=n)

    print(f"Results from collection.query: {results}")

    movie_titles = results["documents"][0]
    movie_imdb_ids = results["ids"][0]

    movies = [
        {
            "id": movie_id,
            "title": movie_title,
        }
        for movie_title, movie_id in zip(movie_titles, movie_imdb_ids)
    ]
    # movies = [
    #     {
    #         "id": res["document"]["ID"],
    #         "title": res["document"]["Title"],
    #         "year": str(res["document"]["Year"]),
    #         "runtime": res["document"]["Run Time"],
    #         "rating": res["document"]["Rating"],
    #         "votes": res["document"]["Votes"],
    #         "metascore": res["document"]["MetaScore"],
    #         "gross": res["document"]["Gross"],
    #         "genre": res["document"]["Genre"],
    #         "certification": res["document"]["Certification"],
    #         "director": res["document"]["Director"],
    #         "stars": res["document"]["Stars"],
    #         "description": res["document"]["Description"],
    #         "plot_keywords": res["document"]["Plot Keywords"],
    #         "imageUrl": res["document"]["Image URL"],
    #     }
    #     for res in results
    # ]
    print("movies")  

    return movies, movie_titles, movie_imdb_ids


def moviebot_chat(user_msg):
    try:
        print("Received message:", user_msg)
        movies = []  # Initialize movies list
        messages = [
            {
                "role": "system",
                "content": "You are a conversational chatbot. Your personality is: helpful, kind, and friendly.",
            }
        ]
        messages.append({"role": "user", "content": user_msg})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        print(f"Full response: {response}")

        # Convert OpenAIObject to dictionary and extract content
        message_dict = response.choices[0].message.to_dict()
        bot_msg = message_dict["content"]

        keywords = ["recommend", "suggest", "movie like", "similar to"]
        is_recommendation_query = any(
            keyword in user_msg.lower() for keyword in keywords
        )
        if is_recommendation_query:
            print("Detected movie recommendation query...")
            movies, _, movie_imdb_ids = search_movies(user_msg, n=6)
            print(f"Movies after search_movies: {movies}")

            # Updated to include hidden IMDb ID in the response
            movie_titles = [movie["title"] for movie in movies]
            bot_msg += f" Based on your request, I also suggest you check out: {', '.join(movie_titles)}."

            print(f"Suggested movies: {', '.join(movie_titles)}")

        else:
            print("Standard query detected (No movie recommendation).")

        return {"bot_msg": bot_msg, "recommended_movies": movies}

    except Exception as error:
        print("Error in moviebot_chat:", error)
        return {
            "bot_msg": "Sorry, something went wrong on my end.",
            "recommended_movies": [],
        }

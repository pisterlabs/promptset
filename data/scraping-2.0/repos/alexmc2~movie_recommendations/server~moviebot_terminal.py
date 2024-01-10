import argparse

import chromadb
import openai
from dotenv import dotenv_values

from embeddings import get_embedding

# Load environment variables
config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

# Setup Chroma client for local access
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "movie_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name)



def insert_movie(movie_data):
    # Generate an embedding for the movie description
    movie_embedding = get_embedding(
        movie_data["Description"], engine="text-embedding-ada-002"
    )

    # Prepare the document to be inserted
    document = {"document": movie_data, "embedding": movie_embedding.tolist()}

    # Insert the document into the collection
    collection.insert(document)


# Admin function to insert movies into the database
def get_movie_input():
    movie_data = {
        "Title": input("Enter movie title: "),
        "Year": input("Enter movie year: "),
        "Run Time": input("Enter movie runtime (e.g., '120 min'): "),
        "Rating": input("Enter movie rating (e.g., '8.5/10'): "),
        "Votes": input("Enter number of votes (e.g., '500,000'): "),
        "MetaScore": input("Enter movie metascore (e.g., '75/100'): "),
        "Gross": input("Enter movie gross earnings (e.g., '$500M'): "),
        "Genre": input("Enter movie genre (e.g., 'Drama, Romance'): "),
        "Certification": input("Enter movie certification (e.g., 'PG-13'): "),
        "Director": input("Enter movie director: "),
        "Stars": input("Enter main stars (e.g., 'Leonardo DiCaprio, Kate Winslet'): "),
        "Description": input("Enter movie description: "),
        "Plot Keywords": input("Enter plot keywords (comma-separated): "),
    }
    return movie_data


def search_movies(query_description, n=3):
    query_embedding = get_embedding(query_description, engine="text-embedding-ada-002")
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n)
    movies = [
        {
            "title": res["document"]["Title"],
            "year": str(res["document"]["Year"]),
            "runtime": res["document"]["Run Time"],
            "rating": res["document"]["Rating"],
            "votes": res["document"]["Votes"],
            "metascore": res["document"]["MetaScore"],
            "gross": res["document"]["Gross"],
            "genre": res["document"]["Genre"],
            "certification": res["document"]["Certification"],
            "director": res["document"]["Director"],
            "stars": res["document"]["Stars"],
            "description": res["document"]["Description"],
            "plot_keywords": res["document"]["Plot Keywords"],
        }
        for res in results
    ]
    return movies


def main():
    parser = argparse.ArgumentParser(
        description="Movie recommendation chatbot with GPT-3.5-turbo and Embeddings"
    )
    parser.add_argument(
        "--personality",
        type=str,
        help="A brief summary of the chatbot's personality",
        default="friendly and helpful",
    )
    args = parser.parse_args()
    initial_prompt = (
        f"You are a conversational chatbot. Your personality is: {args.personality}"
    )
    messages = [{"role": "system", "content": initial_prompt}]

    while True:
        try:
            user_msg = input("\033[1m\033[34mYou: \033[0m")

            # Check if the user wants to insert a new movie
            if user_msg.lower() == "insert movie":
                movie_data = get_movie_input()
                insert_movie(movie_data)
                print("\033[1m\033[31mMoviebot: \033[0mMovie inserted successfully!")
                continue

            messages.append({"role": "user", "content": user_msg})
            streamed_responses = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages, stream=True
            )

            print(
                "\033[1m\033[31mMoviebot: \033[0m", end="", flush=True
            )  # Print the prefix once

            for res in streamed_responses:
                chunk = res["choices"][0]["delta"]
                if "role" in chunk and "content" in chunk:
                    messages.append(chunk)
                if "content" in res["choices"][0]["delta"]:
                    if "search_movies:" in chunk["content"]:
                        query = chunk["content"].split("search_movies:")[-1].strip()
                        recommended_movies = search_movies(query, n=3)
                        for movie in recommended_movies:
                            print(
                                f"\033[1m\033[31mTitle:\033[0m {movie['title']} ({movie['year']})"
                            )
                            print(f"\033[1m\033[31mRun Time:\033[0m {movie['runtime']}")
                            print(f"\033[1m\033[31mRating:\033[0m {movie['rating']}")
                            print(f"\033[1m\033[31mVotes:\033[0m {movie['votes']}")
                            print(
                                f"\033[1m\033[31mMetaScore:\033[0m {movie['metascore']}"
                            )
                            print(f"\033[1m\033[31mGross:\033[0m {movie['gross']}")
                            print(f"\033[1m\033[31mGenre:\033[0m {movie['genre']}")
                            print(
                                f"\033[1m\033[31mCertification:\033[0m {movie['certification']}"
                            )
                            print(
                                f"\033[1m\033[31mDirector:\033[0m {movie['director']}"
                            )
                            print(f"\033[1m\033[31mStars:\033[0m {movie['stars']}")
                            print(
                                f"\033[1m\033[31mDescription:\033[0m {movie['description']}"
                            )
                            print(
                                f"\033[1m\033[31mPlot Keywords:\033[0m {movie['plot_keywords']}"
                            )
                            print()

                    else:
                        print(res["choices"][0]["delta"]["content"], end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()

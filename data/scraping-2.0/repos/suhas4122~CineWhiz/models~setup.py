from .movie import Movie, session
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from ..config import *
import json
from tqdm import tqdm
from time import sleep

movies = []
with open(JSON_PATH, "r") as f:
    data = f.read()
    movies.extend(json.loads(data)["items"])


def create_text(movie: Movie):
    return f"""
Title: {movie.name}
Description: {movie.description}
Review: {movie.review}
"""

def add(movies, vectorstore: Redis, embeddings: OpenAIEmbeddings):
    print("Adding to SQL database")
    texts = []
    metadatas = []
    titles = set()
    for movie_dict in movies:
        if movie_dict["name"] in titles:
            continue
        titles.add(movie_dict["name"])
        movie = Movie(
            name= movie_dict["name"],
            url= movie_dict["url"],
            poster = movie_dict["poster"],
            description = movie_dict["description"],
            review = movie_dict["review"]["reviewBody"],
            rating_count = movie_dict["rating"]["ratingCount"],
            rating_value = movie_dict["rating"]["ratingValue"],
            content_rating = movie_dict["contentRating"],
            genre = movie_dict["genre"],
            date = movie_dict["datePublished"],
            keywords = movie_dict["keywords"],
            duration = movie_dict["duration"],
            actors = movie_dict["actor"],
            creators = movie_dict["creator"],
            director = movie_dict["director"])

        movie.add(session=session)
        texts.append(create_text(movie))
        metadatas.append({"uuid": movie.id})
        
    print("Adding to vectorstore")
    embedding_texts = embeddings.embed_documents(texts)
    vectorstore.add_texts(texts=texts,embeddings=embedding_texts,metadatas=metadatas)

def main():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    redis = Redis(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        embedding_function=embeddings.embed_query,
    )
    redis._create_index()
    redis = Redis.from_existing_index(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        embedding=embeddings,
    )
    batch_size = 300
    for i in range(0,len(movies),batch_size):
        print("Batch ", i)
        batch = movies[i:i+batch_size]
        add(batch, redis, embeddings)
        sleep(20) # to avoid rate limit
        print("Batch done")

if __name__ == "__main__":
    main()
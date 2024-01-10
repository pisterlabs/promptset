import os.path
from pathlib import Path
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from PIL import Image
import numpy as np
from db import setup_db
import requests
from bs4 import BeautifulSoup
import random

vector_db = setup_db()

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=Path("prompts/user_emotions.prompt").read_text(),
)

llm = ChatOpenAI(temperature=0.3)

chain = LLMChain(llm=llm, prompt=prompt)


def get_recommendation(user_emotions: str, k=3):
    recommendation = vector_db.similarity_search(query=user_emotions, k=k)

    return recommendation


def download_cover(movie_title):
    file_extension = '.jpeg'
    image_path = f"cover_images/{movie_title}_cover{file_extension}"
    if os.path.exists(image_path):
        return

    base_url = "https://www.google.com/search"
    params = {
        "q": f"{movie_title} movie poster",
        "tbm": "isch"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    # Perform the Google search
    response = requests.get(base_url, params=params, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    images = soup.find_all('img')
    image_url = None

    for img in images:
        url = img.get('src')
        if url and url.startswith("http"):
            image_url = url
            break

    if image_url:
        image_data = requests.get(image_url).content

        with open(image_path, 'wb') as f:
            f.write(image_data)

        return image_path
    else:
        print("No image found.")
        return None


def process_input(user_input: str):
    def post_process_genres(genres: str):
        genres = genres.replace('\'', '').replace('[', '').replace(']', '')
        return genres

    res = chain.run(user_input=user_input)
    user_emotions = res.split('emotions: ')[1]
    recommendation = get_recommendation(user_emotions)

    # Pick a random element to make it more stochastic
    recommendation = random.choice(recommendation)

    title = recommendation.metadata["title"]
    description = recommendation.metadata["description"]
    genres = post_process_genres(recommendation.metadata["genres"])
    emojis = recommendation.metadata["emojis"]

    download_cover(title)
    cover = np.array(Image.open(f'cover_images/{title}_cover.jpeg'))

    return cover, title, description, genres, emojis

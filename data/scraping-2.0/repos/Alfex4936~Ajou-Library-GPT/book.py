import asyncio
import os

import openai
from aiohttp import ClientSession
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load .env
load_dotenv()

api_key = os.environ.get("API_KEY")

openai.api_key = api_key


async def fetch_books(query, max_results=10):
    print(f"Searching for {query}...")
    async with ClientSession() as session:
        url = f"https://library.ajou.ac.kr/pyxis-api/1/collections/1/search?all=1|k|a|{query}&facet=false&max={max_results}"
        async with session.get(url) as response:
            data = await response.json()
            return data["data"]["list"]


async def get_rent_status_and_locations(book_id):
    async with ClientSession() as session:
        url = f"https://library.ajou.ac.kr/pyxis-api/1/biblios/{book_id}/items"
        async with session.get(url) as response:
            data = await response.json()
            items = data["data"]

        rent_status = {}
        for key in items:
            for item in items[key]:
                location_name = item["location"]["name"]
                circulation_state = item.get("circulationState")
                if circulation_state:
                    is_charged = circulation_state.get("isCharged")
                    is_rentable = (
                        is_charged is False if is_charged is not None else False
                    )
                else:
                    is_rentable = False

                if location_name not in rent_status:
                    rent_status[location_name] = is_rentable
                else:
                    rent_status[location_name] = (
                        rent_status[location_name] or is_rentable
                    )

        return rent_status


def recommend_books(
    student_embedding, book_embeddings, book_data, top_k=5, similarity_threshold=0.4
):
    similarities = [
        (book_id, cosine_similarity([student_embedding], [book_embedding]))
        for book_id, book_embedding in book_embeddings.items()
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    top_k_books = []
    for book_id, similarity in similarities:
        if similarity >= similarity_threshold and len(top_k_books) < top_k:
            top_k_books.append((book_id, book_data[book_id]))

    return top_k_books


def generate_query(interest):
    prompt = f"Generate a bilingual search query (Korean and English) for the following interest: {interest}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trained to generate search queries for book titles based on user prompts. Your goal is to return a list of unique keywords that are most relevant to the user's interests, specifically focusing on higher education level material. Make sure each keyword is relevant to the topic of interest by combining the topic keyword with other relevant keywords. For example, if the user inputs 'Want to learn psychology from scratch', return a comma-separated string of keywords like 'Psychology, Psychology Basics, Psychology Core Concepts, Understanding Psychology, Introduction to Psychology'. Must contain at least the core keyword, in there it was Psychology (심리학)",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        presence_penalty=0.8,
    )

    query = response["choices"][0]["message"]["content"].strip()
    return query


async def main():
    interest = input("도서관 검색: ").strip()

    # Initialize the sentence transformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Example: using GPT to generate a query based on the student's interest
    # interest = "기계학습 공부"
    keywords = [keyword.strip() for keyword in generate_query(interest).split(",")]
    print("GPT-4가 생성한 쿼리:", keywords)

    # Fetch and merge book data for each keyword
    all_books = []
    fetch_book_tasks = [fetch_books(keyword) for keyword in keywords]
    all_books_results = await asyncio.gather(*fetch_book_tasks)

    for books in all_books_results:
        all_books.extend(books)

    # Generate embeddings for each book
    book_embeddings = {
        f"book_id_{book['id']}": model.encode(
            f"{book['titleStatement']} by {book['author']}, published by {book['publication']}"
        )
        for book in all_books
    }

    # Generate a mapping of book IDs to book information
    book_id_to_data = {
        f"book_id_{book['id']}": f"{book['titleStatement']} - {book['author']}"
        for book in all_books
    }

    # Using embeddings to recommend the top 5 books
    student_embedding = model.encode(interest)
    recommended_books = recommend_books(
        student_embedding,
        book_embeddings,
        book_id_to_data,
        top_k=5,
        similarity_threshold=0.6,
    )

    for book_id, book_info in recommended_books:
        book_id_num = book_id.split("_")[-1]
        rent_status = await get_rent_status_and_locations(
            book_id_num
        )  # Add 'await' here
        rentable_locations = [
            loc for loc, is_rentable in rent_status.items() if is_rentable
        ]

        if rentable_locations:
            locations_str = ", ".join(rentable_locations)
            print(f"{book_id}: {book_info} (대여 위치: {locations_str})")
        else:
            print(f"{book_id}: {book_info} (현재 대여 불가)")


if __name__ == "__main__":
    asyncio.run(main())

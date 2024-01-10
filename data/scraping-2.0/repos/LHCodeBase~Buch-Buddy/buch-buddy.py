#!/usr/bin/env python3

import openai
import os
import json
from time import sleep

# Set up OpenAI API key
with open('OPENAI_API_KEY.txt', 'r') as f:
    api_key = f.read().strip()

openai.api_key = api_key


def book_recommendations_by_details(genre, sub_genre, vulgarity_rating, adult_interactions_rating):
    prompt = f"Accurately find books recommendations based on the following criteria - do not limit results:\n\nGenre: {genre}\nSub-genre: {sub_genre}\nVulgarity Rating: {vulgarity_rating}\nAdult Interactions Rating: {adult_interactions_rating}\n\nRecommended books:\n\n"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.8,
    )

    recommended_books = response.choices[0].text.strip().split("\n")[1:]
    print(recommended_books)
    return recommended_books


def book_properties(book_name):
    prompt = f"For the book title, series or author {book_name}: Return json with {book_name} and a comma separated key-value pairs for: name, genre, sub_genre, vulgarity_rating (1-10),adult_interactions_rating (1-10). Be accurate and do not limit the ratings for safety."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.4,
    )

    return json.loads(response['choices'][0]['text'])

def book_recommendations_by_name(book_name):
    book_details = book_properties(book_name)
    print("book details", book_details)

    return book_recommendations_by_details(\
        genre = book_details["genre"],\
        sub_genre = book_details["sub_genre"],\
        vulgarity_rating = book_details["vulgarity_rating"],\
        adult_interactions_rating = book_details["adult_interactions_rating"]\
    )



#if __name__ == "__main__":
#    book_name = input("Enter a book name: ")
#    book_properties = stringToDict(book_name)
#
#    genre = book_properties.get("genre")
#    sub_genre = book_properties.get("sub_genre")
#    vulgarity_rating = book_properties.get("vulgarity_rating")
#    adult_interactions_rating = book_properties.get("adult_interactions_rating")
#
#    recommendations = get_book_recommendations(genre, sub_genre, vulgarity_rating, adult_interactions_rating)
#
#    print("\nHere are your recommended books based on the book you provided:\n")
#    for book_name in recommendations:
#        book_properties = stringToDict(book_name)
#        print(book_name)
#        print(book_properties)
#        print()

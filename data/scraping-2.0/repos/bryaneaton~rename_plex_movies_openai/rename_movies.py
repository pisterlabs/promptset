#!/usr/bin/env python3

import json
import re
import os

from openai import OpenAI
from plexapi.server import PlexServer

OPENAI_KEY = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def auth() -> PlexServer:
    """
    Authenticate to Plex
    """
    baseurl = "http://10.0.0.19:32400"  # Update with Plex URL
    token = "PLEXTOKEN"  # Add Plex Token
    return PlexServer(baseurl, token)


def find_strings_with_multiple_periods(text):
    """
    Find strings with multiple periods

    Args:
        text (str): The text to search
    """
    pattern = re.compile(r"(\w+\.\w+\.\w+)")
    return pattern.findall(text)


def get_movie_library(plex: PlexServer) -> list:
    """
    Get the movie library
    """
    return plex.library.section("Movies")


def prompt_movie_title(borked_title: str) -> dict:
    prompt = (
        f"Extract the year and title from this string as a valid json :{borked_title}."
        + "  I want the returned json to have lower case keys."
        + "  I want the returned json to have the keys 'title' and 'year'."
        + "  I want year to be an integer, always, and title is always a string"
        + "  I want the title of the returned json to have proper casing"
    )

    # Make a request to the GPT-3.5 model
    response = client.completions.create(
        model="text-davinci-003", prompt=prompt, max_tokens=150
    )
    result = (response.choices[0].text).replace("\n", "")
    print(result)
    try:
        return json.loads(result)
    except json.decoder.JSONDecodeError:
        return None


def open_file():
    with open("new_titles.json", "r") as f:
        dat = json.load(f)
    return dat


if __name__ == "__main__":
    plex = auth()
    movie_library = get_movie_library(plex)
    new_titles = []
    for movie in movie_library.search():
        if find_strings_with_multiple_periods(movie.title):
            new_title = prompt_movie_title(movie.title)
            try:
                dat = {
                    "old_title": movie.title,
                    "new_title": new_title["title"],
                    "year": new_title["year"],
                }
                if new_title:
                    new_titles.append(dat)
                    update = {
                        "title.value": new_title["title"],
                        "year.value": new_title["year"],
                    }
                    movie.edit(**update)
                    movie.reload()
                    print(movie.title)
            except TypeError:
                print("TypeError with returned data:", new_title)

    # Save the data
    with open("new_titles.json", "w") as f:
        json.dump(new_titles, f, indent=4)

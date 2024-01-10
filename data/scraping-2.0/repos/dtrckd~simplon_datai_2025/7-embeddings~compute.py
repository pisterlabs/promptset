#!/usr/bin/python

import json
import os
from pprint import pprint
from typing import List, Union

import numpy as np
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup

openai.api_key = os.environ["OPENAI_API_KEY"]


url_first = "https://en.wikipedia.org/wiki/Machine_learning"
title_first = "Machine learning"


def scrape_urls(url_first: str) -> dict:
    """
    Returns: a list of dict like this {"url":str, "title": str}
    """
    domain = url_first.split("/")[2]
    html_content = requests.get(url_first).text

    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for link in soup.find_all("a"):
        url = link.get("href")
        # Filter bad url
        # - anchor
        # - stay in english / original domain

        if not url or not url.startswith("/wiki"):
            continue

        title = link.get("title")
        if not title:
            continue

        links.append({"url": domain + url, "title": title})

    df = pd.DataFrame(links)
    df = df.drop_duplicates("url")
    links = sorted(df.to_dict("records"), key=lambda x: x["url"])
    # urls = sorted(list(set(urls)))

    return links


def get_embedding(text: str) -> List[float]:
    try:
        data = openai.Embedding.create(model="text-embedding-ada-002", input=text, encoding_format="float")
        embedding = data["data"][0]["embedding"]
    except:
        print("open failed once, retrying...")
        data = openai.Embedding.create(model="text-embedding-ada-002", input=text, encoding_format="float")
        embedding = data["data"][0]["embedding"]

    return embedding


def extract_content(url: str) -> str:
    pass
    # html_content = requests.get(url).text

    # soup = BeautifulSoup(html_content, "html.parser")
    # contents = []
    # for link in soup.find_all("a"):


def compute_sim(vector_first: List[float], vectors: List[list]) -> List[float]:
    v1 = np.array(vector_first)
    v2s = np.array(vectors)
    sims = v2s.dot(v1)
    return sims


if __name__ == "__main__":
    # Extract URL from a webpage
    links = scrape_urls(url_first)
    # pprint(links)


    # Extract text content for each URL
    # NOTE: we extact the content (title etc) in the scraping function for now
    # the following do nothin.
    corpus = []
    for link in links:
        corpus.append(extract_content(link["url"]))

    # Get embeddings for eacsh text
    vectors = []
    for link in links:
        vectors.append(get_embedding(link["title"]))
        print(".", end="", flush=True)

    # compute similarities
    vector_first = get_embedding(title_first)
    similarities = compute_sim(vector_first, vectors)

    # Update links dicts
    for i, sim in enumerate(similarities):
        links[i]["sim"] = sim

    # Save the results
    with open("results.json", "w") as f:
        json.dump(links, f)

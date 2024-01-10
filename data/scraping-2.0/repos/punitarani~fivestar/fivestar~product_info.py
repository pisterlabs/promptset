"""fivestar.product_info.py"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import pandas as pd
from langchain.docstore.document import Document
from tenacity import retry, stop_after_attempt, wait_fixed

from fivestar.store import vectorstore

AMAZON_PRODUCT_BASE_URL = "https://www.amazon.com/dp/"
DATA_DIR = Path(__file__).parent.parent.joinpath("data")

loaded_product_info = set()
loaded_product_vectors = set()


async def get_product_info(product_id: str) -> dict:
    """
    Get Product Information from Amazon.
    :param product_id: Product ID to lookup.
    :return: Product information.
    """
    url = "https://amazon23.p.rapidapi.com/product-details"
    querystring = {"asin": product_id, "country": "US"}

    headers = {
        "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
        "X-RapidAPI-Host": "amazon23.p.rapidapi.com"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=querystring, timeout=30)
        data = json.loads(response.content).get("result", [])[0]
        info = {
            "title": data.get("title", product_id),
            "description": data.get("description", ""),
            "features": data.get("feature_bullets", []),
        }
    except Exception as error:
        error_msg = f"Unable to get product info for {product_id}. {error}"
        print(error_msg)
        raise ValueError(error_msg) from error

    with open(DATA_DIR.joinpath(f"info/{product_id}.json"), "w") as f:
        json.dump(info, f)
    return info


async def load_product_info(product_id: str) -> dict:
    """
    Load Product Information from a json file.
    :param product_id: Product ID to lookup.
    :return: Product information.
    """
    try:
        with open(DATA_DIR.joinpath(f"info/{product_id}.json"), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = await get_product_info(product_id)

    if product_id not in loaded_product_info:
        vectorstore.add_documents([
            Document(page_content=data["title"]),
            Document(page_content=data["description"]),
            *[Document(page_content=feature) for feature in data["features"]]
        ])
        loaded_product_info.add(product_id)

    return data


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def get_product_reviews(product_id: str, num_pages: int = 6) -> pd.DataFrame:
    """
    Get product reviews from Amazon.
    :param product_id: Product ID to lookup.
    :param num_pages: Number of pages to get. Each page contains 10 reviews.
    :return: DataFrame with the product reviews.

    Note: Saves the product reviews to a csv file in /data/reviews/{product_id}.csv
    """
    tasks = []
    num_half_pages = num_pages // 2
    for page in range(1, num_half_pages + 1):
        tasks.append(_get_product_reviews_page(product_id, page, "helpful"))
        tasks.append(_get_product_reviews_page(product_id, page + num_half_pages, "recent"))

    reviews = await asyncio.gather(*tasks)
    reviews_flat = [item for sublist in reviews for item in sublist.get("result", [])]
    print(f"Got {len(reviews_flat)} reviews for {product_id}")

    # map the response to your desired DataFrame structure
    df = pd.DataFrame(reviews_flat)
    df = df.rename(columns={
        "id": "review_id",
        "rating": "star_rating",
        "title": "review_headline",
        "review": "review_body"
    })
    df = df[["review_id", "star_rating", "review_headline", "review_body"]]

    # save the reviews to a csv file
    df.to_csv(DATA_DIR.joinpath(f"reviews/{product_id}.csv"), index=False)

    return df


async def load_product_reviews(product_id: str) -> pd.DataFrame:
    """
    Load product reviews from a csv file.
    :param product_id: Product ID to lookup.
    :return: DataFrame with the product reviews.
    """
    try:
        df = pd.read_csv(DATA_DIR.joinpath(f"reviews/{product_id}.csv"))
    except FileNotFoundError:
        df = await get_product_reviews(product_id)

    # Create docs from the dataframe and store them in the vectorstore
    if product_id not in loaded_product_vectors:
        docs = [
            Document(page_content=review_body, metadata={"product_id": product_id})
            for review_body in df["review_body"].tolist()
        ]
        vectorstore.add_documents(docs)
        loaded_product_vectors.add(product_id)

    return df


async def _get_product_reviews_page(product_id: str, page: int = 1, sort_by: str = "helpful") -> dict:
    """
    Get a page of product reviews from Amazon.
    :param product_id: Product ID to lookup.
    :param page: Page number to get. Each page contains 10 reviews.
    :param sort_by: Sort by helpful or recent rating.
    :return: Dictionary with the product reviews.
    """
    url = "https://amazon23.p.rapidapi.com/reviews"
    querystring = {"asin": product_id, "sort_by": sort_by, "page": str(page), "country": "US"}

    headers = {
        "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
        "X-RapidAPI-Host": "amazon23.p.rapidapi.com"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=querystring, timeout=30)
    return json.loads(response.content)

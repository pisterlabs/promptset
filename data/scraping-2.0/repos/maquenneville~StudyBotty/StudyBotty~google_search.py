# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:06:55 2023

@author: marca
"""


import requests
from bs4 import BeautifulSoup
import re
from openai_pinecone_tools import *
from ingester import chunk_text
from nltk.tokenize import sent_tokenize
from urllib.parse import unquote, urlparse, urljoin
import wikipedia
from typing import Set
from googleapiclient.discovery import build
import os


def sanitize_url(url: str) -> str:
    """Sanitize the URL
    Args:
        url (str): The URL to sanitize
    Returns:
        str: The sanitized URL
    """
    return urljoin(url, urlparse(url).path)


def get_text_from_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.extract()

        text = soup.get_text()

        # Remove leading and trailing spaces on each line
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a single line
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop empty lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def get_wiki_page(title: str):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page, False
    except wikipedia.DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False), True
    except Exception:
        return None, False


def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i]) + "."

    return long_text


discard_categories = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
    "ISBN",
]


def extract_wiki_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
) -> str:
    if len(wiki_text) == 0:
        return []

    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont) + 4)]

    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if num_equals <= keep_group_level:
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    outputs += [
        (title, h, c, t)
        if t < max_len
        else (title, h, reduce_long(c, max_len=max_len), max_len)
        for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)
    ]

    content = [output[2] for output in outputs]

    return "\n".join(content)


def is_advertisement(link):
    ad_pattern = re.compile(r"^https?://(www\.)?google\..*/aclk\?.*$", re.IGNORECASE)
    return bool(ad_pattern.match(link))


def get_search_links(
    search_phrase, num_results=3, api_key=GOOGLE_API_KEY, cx=GOOGLE_ID
):
    approved_links = []

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        start_index = 1

        while start_index <= num_results:
            response = (
                service.cse().list(q=search_phrase, cx=cx, start=start_index).execute()
            )

            if "items" not in response:
                break

            for item in response["items"]:
                if "link" in item:
                    approved_links.append(item["link"])

                if len(approved_links) >= num_results:
                    break

            start_index += 10

    except Exception as e:
        print(f"Error: {e}")

    return approved_links


def summarize_webpage(url_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are a highly intelligent AI assistant.  Your job is to summarize the text contents of a website, exclusively to be used as context by ChatGPT.  The summary will never be read by a human, only ChatGPT.  Keep the summary as short as possible while maintaining all details.",
        },
    ]

    messages += [
        {"role": "user", "content": "Please provide a summary of this webpage text."},
        {
            "role": "user",
            "content": f"""Web Page Text:
                     {url_text}
                     """,
        },
    ]

    # Generate the summary using the provided generate_response function
    summary = generate_response(messages, temperature=0.0)
    return summary


def compress_summary(summary):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are a very effective and intelligent AI assistant.  You compress inputs for ChatGPT.  Your job is to compress webpage summaries for ChatGPT.  These compressed summaries are not meant to be read by humans, rather they are only meant to be read by ChatGPT and can be completely unintelligible to humans.  Make sure the compression is lossless.",
        },
    ]

    compression_prompt = "Please compress this summary of a conversation."
    messages += [
        {"role": "user", "content": compression_prompt},
        {
            "role": "user",
            "content": f"""Summary to be compressed:
                     {summary}
                     """,
        },
    ]

    # Generate the summary using the provided generate_response function
    compressed_summary = generate_response(messages, temperature=0.2)

    return compressed_summary


def compress_text(text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are a ChatGPT Text Compression Specialist, and the inventor of ChatGPT-Lossless Compression (CLC).   Description: CLC removes filler words, abbreviates common terms, and applies other techniques to losslessly compress text, while maintaining readability for AI language models like ChatGPT.  Example:\nOriginal-\n'The house stood on a slight rise just on the edge of the village. It stood on its own and looked out over a broad spread of West Country farmland. Not a remarkable house by any means—it was about thirty years old, squattish, squarish, made of brick, and had four windows set in the front of a size and proportion which more or less exactly failed to please the eye.'\nCompressed-\n'House std on slight rise @ edge village. Alone, lkd ovr broad West Country farmland. Not remarkable, ~30 yrs old, squattish, squarish, brick, 4 wndws, size+proportion failed 2 pls eye.'",
        },
    ]

    compression_prompt = "Please compress this text."
    messages += [
        {"role": "user", "content": compression_prompt},
        {
            "role": "user",
            "content": f"""Text to be compressed:
                     {text}
                     """,
        },
    ]

    # Generate the summary using the provided generate_response function
    compressed_summary = generate_response(messages, temperature=0.0)

    return compressed_summary


def process_url(url):
    def is_wikipedia_link(url: str) -> bool:
        """
        Check if the given URL is a Wikipedia link.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is a Wikipedia link, False otherwise.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return "wikipedia.org" in domain

    def get_wikipedia_title_from_url(url: str) -> str:
        """
        Extract the title of a Wikipedia page from its URL.

        Args:
            url (str): The URL of the Wikipedia page.

        Returns:
            str: The title of the Wikipedia page.
        """
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split("/")

        if len(path_parts) > 2 and path_parts[1] == "wiki":
            title = unquote(path_parts[2]).replace("_", " ")
            return title
        else:
            return None

    # Sanitize the URL
    sanitized_url = sanitize_url(url)

    if is_wikipedia_link(sanitized_url):
        wiki_title = get_wikipedia_title_from_url(sanitized_url)
        wiki_page, _ = get_wiki_page(wiki_title)

        text = extract_wiki_sections(wiki_page.content, wiki_page.title)
    else:
        # Get the text from the website
        text = get_text_from_website(sanitized_url)

    if not text:
        return None

    max_sum_chunk_len = 4000

    sum_chunks = []
    for start in range(0, len(text), max_sum_chunk_len):
        end = min(start + max_sum_chunk_len, len(text))
        sum_chunks.append(text[start:end])

    compressed = [compress_text(sum_chunk) for sum_chunk in sum_chunks]

    # Create a DataFrame with embeddings
    df = create_embeddings_dataframe(compressed)

    return df


def generate_search_phrase(
    query: str, temperature=0.0, max_tokens=15, frequency_penalty=0
):
    # Set up the messages for the ChatGPT API call
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my Google Search Phrase Assistant. Your job is to take a query, then create a search phrase intended for use with Google Search.  Ensure the search phrase is structured to get the most relevant results from Google Search.  Only generate one search phrase.",
        },
        {
            "role": "user",
            "content": f"Generate a search phrase for the following query: '{query}'",
        },
    ]

    # Call the generate_response function with the specified messages and other parameters
    search_phrase = generate_response(
        messages,
        temperature=temperature,
        n=1,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )

    return search_phrase.strip('"')


def google_search_agent(
    query,
    namespace=GOOGLE_NAMESPACE,
    index=PINECONE_INDEX,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
):
    search_phrase = generate_search_phrase(query)

    search_links = get_search_links(search_phrase)

    for link in search_links:
        link_df = process_url(link)
        store_embeddings_in_pinecone(dataframe=link_df, namespace=GOOGLE_NAMESPACE)

    context = fetch_context_from_pinecone(query, namespace=GOOGLE_NAMESPACE)

    return context


# print(google_search_agent("What is the world's largest diesel engine?"))

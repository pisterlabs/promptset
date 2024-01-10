from persistence import rPost
from retry import retry
from bz2 import compress, decompress
from tiktoken import get_encoding
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from fitz import open as open_pdf
from os import getenv
from json import dumps
import requests
import openai

# Set to a global variable to avoid calling the function every time.
enc = get_encoding("cl100k_base")

# We define OpenAI's API key and endpoint.
openai.api_key = getenv("AZURE_AI_API_KEY")
openai.api_base = getenv("AZURE_AI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = getenv("AZURE_AI_VERSION")

# We define the proxies to use.
proxies = {
    'http': getenv("PROXY_URL_USA"),
    'https': getenv("PROXY_URL_USA"),
}

# Constants
# The maximum number of tokens we will use to compute embeddings.
MAX_TOKENS = 512
DIMENSIONS = 1536  # The number of dimensions of the embeddings.
MODEL_ID = "text-embedding-ada-002"  # The ID of the model to use.


def add_embeddings_redis(id: str):
    """
    Upsert the embeddings of a post to Redis.
    """

    id = "hn:{}".format(id)
    res = rPost.hget(id, "url")
    if res is None:
        raise Exception("URL is empty.")
        return
    res = res.decode("utf-8")
    if res == "":
        raise Exception("URL is empty.")

    embeddings = compute_embeddings(res)

    if len(embeddings) == 0:
        raise Exception("Embeddings are empty.")

    stringifiedJSON = dumps(embeddings)

    compressedJSON = compress(stringifiedJSON.encode("utf-8"))

    rPost.hset(id, "embeddings", compressedJSON)


def compute_embeddings(url: str) -> list[float]:
    """
    Compute the embeddings of a URL from the text of the article.
    First, we get the text of the article.
    Then, we shrink the text to MAX_TOKENS tokens.
    Finally, we compute the embeddings of the text.

    The dimension for text-embedding-ada-002 is 1536.

    Args:
        url (str): The URL of the article.

    Returns:
        list[float]: The embeddings of the article.
    """
    text = get_text(url)
    text = get_text_truncated_tokenized(text, MAX_TOKENS)

    if (len(text) == 0):
        raise Exception("Text extracted is empty.")

    # We compute the embeddings.
    response = openai.Embedding.create(input=text, model=MODEL_ID, deployment_id=getenv("AZURE_DEPLOYMENT_ID"))[
        'data'][0]['embedding']

    return response


def get_text(url: str) -> str:
    """
    Sort the type of URL and call the appropriate function to extract the text.

    Args:
        url (str): The URL of the article.
    """
    # Because all websites are different, we need to use different functions to
    # extract the text.
    # To know which function to use, we parse the URL.

    parsed = urlparse(url)

    if (parsed.hostname == "www.youtube.com" or parsed.hostname == "youtube.com" or parsed.hostname == "youtu.be"):
        return get_text_YouTube(url)

    # We check if the URL is a PDF.
    # To do so, we send a HEAD request to the URL and check the Content-Type.
    # If the Content-Type is application/pdf, we use the function get_text_pdf.
    # I prefer this solution to checking the extension because the extension
    # can be wrong (e.g. a PDF served without an extension).

    # We send a HEAD request to the URL.
    response = requests.head(url, allow_redirects=True, proxies=proxies)

    if response.status_code == 404 or response.status_code >= 500:
        raise Exception(
            "Status code is not 200: {}".format(response.status_code))

    if "Content-Type" not in response.headers:
        response.headers["Content-Type"] = "text/html"

    # We check if the Content-Type is application/pdf.
    if response.headers["Content-Type"] == "application/pdf":
        return get_text_pdf(url)

    # We check if the URL is an image.
    matching_types = ["image/jpeg", "image/png",
                      "image/gif", "image/webp", "image/tiff", "image/bmp"]
    if response.headers["Content-Type"] in matching_types:
        # We raise an exception because we can't compute embeddings from an image.
        raise Exception("URL {} is an image. We can't extract text from an image.".format(
            url))

    # Before checking if a URL is an Arxiv article, we check if it's a PDF.
    # get_text_arxiv will modify the URL to get the PDF URL.
    # We don't want to modify the URL if it's already a PDF URL so we check before.

    if (parsed.hostname == "www.arxiv.org" or parsed.hostname == "arxiv.org"):
        return get_text_Arxiv(url)

    # If the URL didn't match any of the previous cases, we hope it's an article.
    return get_text_Article(url)


def get_text_Article(url: str) -> str:
    """
    Fetch the text of an article using Diffbot's Article API.

    Args:
        url (str): The URL of the article.

    Returns:
        str: The text of the article.
    """

    params = {
        "url": url,
        "token": getenv("DIFFBOT_API_KEY"),
    }

    headers = {
        "Accept": "application/json",
    }

    response = requests.get(
        "https://api.diffbot.com/v3/article", params=params, headers=headers, proxies=proxies)

    # We check if the request was successful.
    if (response.status_code != 200):
        raise Exception("Error while fetching the text of the article. Status code: {}".format(
            response.status_code))

    if "error" in response.json():
        raise Exception("Error while fetching the text of the article: {}".format(
            response.json()["error"]))

    data = response.json()["objects"][0]

    # We check if the text is returned by the API.
    if ("text" not in data):
        raise Exception("Error while fetching the text of the article: {}".format(
            data["error"]))

    return data["text"]


def get_text_YouTube(url: str) -> str:

    # ---------------------------- Parse the video ID ---------------------------- #

    video_id = ""
    parsed = urlparse(url)
    path = parsed.path
    if (path.startswith("/watch")):
        # We extract the ID of the video.
        # The ID is the value of the v parameter.
        # For example, if the URL is https://www.youtube.com/watch?v=9bZkp7q19f0, the ID is 9bZkp7q19f0.
        video_id = parsed.query.split("=")[1]
    elif (path.startswith("/embed/")):
        # We extract the ID of the video.
        # The ID is the last part of the path.
        # For example, if the URL is https://www.youtube.com/embed/9bZkp7q19f0, the ID is 9bZkp7q19f0.
        video_id = path.split("/")[-1]
    elif (path.startswith("/v/")):
        # We extract the ID of the video.
        # The ID is the last part of the path.
        # For example, if the URL is https://www.youtube.com/v/9bZkp7q19f0, the ID is 9bZkp7q19f0.
        video_id = path.split("/")[-1]
    elif (path.startswith("/playlist?list=")):
        # We raise an exception because we can't compute embeddings from a playlist.
        return get_text_Article(url)
    elif parsed.hostname == "youtu.be":
        # We extract the ID of the video.
        # The ID is the value of the path.
        # For example, if the URL is https://youtu.be/9bZkp7q19f0, the ID is 9bZkp7q19f0.
        video_id = path[1:]
    else:
        # We raise an exception because we can't compute embeddings from a channel.
        raise Exception("We can't extract text from this url: {}".format(url))

    # ---------------------------- Get the captions ---------------------------- #
    captions = YouTubeTranscriptApi.get_transcript(
        video_id, proxies=proxies, languages=["en"])

    # We extract the text from the captions.
    text = ""
    for caption in captions:
        text += caption["text"] + " "

    return text


def get_text_pdf(url: str) -> str:

    # We download the PDF.
    # We use the stream parameter to avoid loading the whole PDF in memory.
    # We use the timeout parameter to avoid waiting too long for the PDF.
    response = requests.get(url, stream=True, timeout=15)

    # We open the PDF.
    # We use the context manager to close the PDF automatically.
    with open_pdf(stream=response.content, filetype="pdf") as pdf:
        # We get the text of the PDF.
        text = ""
        for page in pdf:
            text += page.get_text()

        # We return the text.
        return text


def get_text_Arxiv(url: str) -> str:
    # We extract the ID of the article.
    # The ID is the last part of the path.
    # For example, if the URL is https://arxiv.org/abs/1702.01715, the ID is 1702.01715.

    parsed = urlparse(url)
    path = parsed.path
    id = path.split("/")[-1]

    # We get the PDF URL.
    url = "https://arxiv.org/pdf/{}.pdf".format(id)
    return get_text_pdf(url)


def get_text_truncated_tokenized(text: str, max_tokens: int) -> str:
    """
    Truncate a text to the desired number of tokens.
    It's to avoid excessive costs when computing embeddings.

    Args:
        text (str): The text to truncate.
        max_tokens (int): The maximum number of tokens in cl100k_base

    """
    # We tokenize the text.
    tokens = enc.encode(text)

    # We truncate the tokens.
    tokens = tokens[:max_tokens]

    # We decode the tokens.
    text = enc.decode(tokens)

    # As stated here: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#embeddings
    # It's best to replace newlines with spaces.
    text = text.replace("\n", " ")

    return text


if __name__ == "__main__":
    # compute_embeddings("https://www.youtube.com/watch?v=IPBSB1HLNLo")
    # compute_embeddings("https://arxiv.org/abs/1702.01715")
    # compute_embeddings("https://arxiv.org/pdf/2305.18179.pdf")
    # compute_embeddings("https://bitcoin.org/bitcoin.pdf")
    # compute_embeddings("https://python-rq.org/docs/workers/")

    add_embeddings_redis("132117")

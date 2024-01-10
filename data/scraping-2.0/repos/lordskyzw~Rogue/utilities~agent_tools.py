import os
from tweepy import Client
from serpapi import GoogleSearch
import requests
import base64
from openai import OpenAI
from pygwan import WhatsApp

token = os.environ.get("WHATSAPP_ACCESS_TOKEN")
phone_number_id = os.environ.get("PHONE_NUMBER_ID")
openai_api_key = str(os.environ.get("OPENAI_API_KEY"))
messenger = WhatsApp(token=token, phone_number_id=phone_number_id)
oai = OpenAI(api_key=openai_api_key)

class ChiefTwit(Client):
    def __init__(self):
        self.consumer_key = os.environ.get("twitter_consumer_key")
        self.consumer_secret = os.environ.get("twitter_consumer_secret")
        self.access_token = os.environ.get("twitter_access_token")
        self.access_token_secret = os.environ.get("twitter_access_token_secret")
        self.client = Client(
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
        )

    def write_tweet(self, text):
        """use when you need to write/send/make a tweet"""
        response = self.client.create_tweet(text=text)
        if response.errors == []:
            return "Successful" 
        else:
            return "Something went wrong"

    def get_tweets(self, username):
        self.client.get_user(username)

    def get_followers(self, username):
        self.client.get_followers(username)

    def get_following(self, username):
        self.client.get_following(username)

    def get_user(self, username):
        self.client.get_user(username)


class SearchProcessor:
    def __init__(self):
        self.api_key = os.environ.get("SERP_API_KEY")
        self.base_url = "https://serpapi.com" 

    def get_search_results(self, query):
        """Method to interact with the API and get the search results."""
        params = {
            "q": query,
            "api_key": self.api_key
        }
        try:
            response = requests.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _process_response(self, res):
        """Process response from the API."""
        if "error" in res:
            raise ValueError(f"Got error from API: {res['error']}")

        toret = "No good search result found"

        if "answer_box" in res:
            answer_box = res["answer_box"][0] if isinstance(res["answer_box"], list) else res["answer_box"]
            if "answer" in answer_box:
                toret = answer_box["answer"]
            elif "snippet" in answer_box:
                toret = answer_box["snippet"]
            elif "snippet_highlighted_words" in answer_box:
                toret = answer_box["snippet_highlighted_words"][0]

        elif "sports_results" in res and "game_spotlight" in res["sports_results"]:
            toret = res["sports_results"]["game_spotlight"]

        elif "shopping_results" in res and res["shopping_results"]:
            toret = res["shopping_results"][:3]

        elif "knowledge_graph" in res and "description" in res["knowledge_graph"]:
            toret = res["knowledge_graph"]["description"]

        elif "organic_results" in res and res["organic_results"]:
            if "snippet" in res["organic_results"][0]:
                toret = res["organic_results"][0]["snippet"]
            elif "link" in res["organic_results"][0]:
                toret = res["organic_results"][0]["link"]

        elif "images_results" in res and res["images_results"]:
            thumbnails = [item["thumbnail"] for item in res["images_results"][:10]]
            toret = thumbnails

        return toret

    def run(self, query):
        """Run query through the API and parse result."""
        return self._process_response(self.get_search_results(query))


def encode_image(image_path):
    '''This function encodes an image into base64'''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image(description: str):
    '''this function should generate an image and return url'''
    res = oai.images.generate(
        prompt=description,
        model="dall-e-3",
        n=1,
        quality="standard",
        style="vivid",
        size="1024x1024",
        response_format="url"
        )
    try:
        url = res.data[0].url
        return url
    except Exception as e:
        return str(e)

def analyze_images_with_captions(image_url: str, caption: str):
    """
    Analyzes images using OpenAI's GPT-4-Vision model and returns the analysis.

    :param image_urls: A list of image URLs to be analyzed.
    :param captions: A list of captions corresponding to the images.
    :return: The response from the OpenAI API.
    """
    if not image_url or not caption:
        raise ValueError("Image and captions cannot be empty")
    
    image_uri = messenger.download_media(media_url=image_url, mime_type="image/jpeg")
    base64_image = encode_image(image_uri)
    
    # Construct the messages payload
    messages = []
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",}
            }
        ]
    }
    messages.append(message)

    # Send the request to OpenAI
    response = oai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300
    )

    return response.choices[0].message.content

def search(query):
    google = GoogleSearch(params_dict={'q': query, 'api_key': os.environ.get('SERP_API_KEY')})
    results = google.get_results()
    return results
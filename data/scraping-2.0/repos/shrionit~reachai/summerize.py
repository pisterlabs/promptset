import requests
from bs4 import BeautifulSoup
import openai
from tools import *
from constants import OPENAI_KEY, URL

openai.api_key = OPENAI_KEY


def scrape_webpage(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text
    except Exception as e:
        print(f"Error scraping the webpage: {str(e)}")
        return None

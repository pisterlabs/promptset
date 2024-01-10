import requests
from bs4 import BeautifulSoup
import openai

from utils.logger import get_logger

logger = get_logger(__name__)


class DataExtractor:

    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_webpage_text(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else ""

        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = "\n".join(paragraphs)
        print(text)

        return title, text

    def summarize(self, title, text):
        prompt = f"nSummarize in 250 words in the language of source:\n{title}\n{text}"
        response = openai.Completion.create(
            engine="text-davinci-003", prompt=prompt, max_tokens=300)

        return response.choices[0].text.strip()  # type: ignore

    def extract_and_summarize(self, url):
        return str(url)
        # try:
        #     title, text = self.get_webpage_text(url)
        #     summary = self.summarize(title, text)
        #     return summary
        # except Exception as e:
        #     log.error(f"Error extracting and summarizing: {e}")
        # return None

from typing import List
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter

def get_url_content(url: str, chunk_size = 1000) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = chunk_size,
        chunk_overlap = 0,
        length_function = len,
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to get url content. Status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = soup.find('h1').text.strip()
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        article_text += paragraph.text + '\n'
    return text_splitter.split_text(article_text)


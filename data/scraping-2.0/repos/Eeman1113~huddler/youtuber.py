import requests
from langchain.document_loaders import YoutubeLoader
from langchain.schema import Document
from bs4 import BeautifulSoup

def get_youtube_video_title(video_url):
    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('meta', property='og:title')
    if title:
        return title['content']
    else:
        return "Title not found"

# Function to fetch captions for a single video
def fetch_youtube_captions(video_url):
    title = get_youtube_video_title(video_url)
    loader = YoutubeLoader.from_youtube_url(video_url)
    docs = loader.load()

    # Prepend the title to the first document's content
    if docs and len(docs) > 0:
        intro_sentence = "This is the title of the video/transcription/conversation: "
        title_content = intro_sentence + title
        docs[0] = Document(page_content=title_content + "\n\n" + docs[0].page_content)

    return docs
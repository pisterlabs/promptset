import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
# list of document loaders for youtube videos
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
# list of document loaders for wikipedia and other websites
from langchain.document_loaders import WikipediaLoader
from langchain.document_loaders import WebBaseLoader

# list of document loaders for Notion
from langchain.document_loaders import NotionDirectoryLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load a PDF document
pdf_loader = PyPDFLoader("/home/vv070/Desktop/LangchainApp/fin_irjmets1687886863.pdf")
page= pdf_loader.load()
# show the number of pages
length = len(page)
print(length)
# show the content of the first page
#content= page[0]
#print(content)
# show the directory of the content
#d= content.metadata
#print(d)

# # load a url from youtube
# url= "https://www.youtube.com/watch?v=aywZrzNaKjs"
# save_path = "/home/vv070/Desktop/LangchainApp/youtube_audio"
# # load the url
# yloader= GenericLoader(
#     YoutubeAudioLoader([url], save_path),
#     OpenAIWhisperParser()
# )
# docs = yloader.load()
# # save the content of the youtube url in a text file
# with open("/home/vv070/Desktop/LangchainApp/youtube_audio.txt", "w") as f:
#     f.write(docs[0].page_content)

# show the content of the url
#d= docs[0].page_content[0:1000]
#print(d)
# install you library
# pip install yt-dlp
# pip install ffmpeg
# pip install pydub

# laod the data from the url like wikipedia or other websites
Web_loader = WebBaseLoader("https://stackoverflow.com/questions/30770155/ffprobe-or-avprobe-not-found-please-install-one")
web_docs = Web_loader.load()
# show the content of the url
d= web_docs[0].page_content[0:1000]
#print(d)
# save the content of the url in a text file
with open("/home/vv070/Desktop/LangchainApp/stackoverflow.txt", "w") as f:
    f.write(web_docs[0].page_content)

# # load the data from wikipedia
# wiki_loader = WikipediaLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
# wiki_docs = wiki_loader.load()
# # show the content of the url
# d= wiki_docs[0].page_content[0:1000]
# print(d)
# # save the full content of the url in a pdf file
# with open("/home/vv070/Desktop/LangchainApp/wikipedia.pdf", "w") as f:
#     f.write(wiki_docs[0].page_content)

# load the data from Notion
#notion_loader = NotionDirectoryLoader("https://www.notion.so/Notion-Test-Page-1-0b1e2f2f1b9e4b6e8b8b8b8b8b8b8b8b")
notion_loader = NotionDirectoryLoader("/home/vv070/Desktop/LangchainApp/docs/hawai.md")
notion_docs = notion_loader.load()

# show the content of the url

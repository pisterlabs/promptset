from langchain.document_loaders import WebBaseLoader
import nest_asyncio

def extract_data():
    urls = ['https://www.daraz.pk/home-improvement-tools/']
    nest_asyncio.apply()
    loader = WebBaseLoader(urls)
    data = loader.aload()
    return data

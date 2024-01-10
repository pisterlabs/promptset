from langchain.document_loaders import TextLoader
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter

class DataExtraction:
    @staticmethod
    def scrape_data(link):
        r = requests.get(link)
        soup = BeautifulSoup(r.content, 'html.parser')
        data = []
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            data.append(paragraph.get_text())
        text_data = '\n'.join(data)
        with open('data.txt', 'w', encoding='utf-8') as file:
            file.write(text_data)
        return 'data.txt'
    def convert_to_docs(input_link):
        loader = TextLoader(DataExtraction.scrape_data(input_link))
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        return docs
    
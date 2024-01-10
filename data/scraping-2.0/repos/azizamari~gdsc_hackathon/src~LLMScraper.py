from langchain_alt import BSHTMLLoader, TextLoader
from langchain.chains.summarize import load_summarize_chain
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import json
load_dotenv()
import os

class LLMScraper:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
        self.test_url = "https://boredgeeksociety.beehiiv.com/p/ai-weekly-digest-14-ai-lowcode-is-a-productivity-game-changer"

    def get_html_content(self, url=test_url):
        headers = os.environ['headers']
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html_content = response.text
            return html_content
        else:
            return 'Error'

    def get_text_content(self, content):
        bshtml_loader = BSHTMLLoader(html_text=content)
        document = bshtml_loader.load()
        text_content = document[0].page_content
        return text_content

    def get_text_structure(self, text):
        loader = TextLoader(text=text)
        docs = loader.load()
        text_splitter = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)

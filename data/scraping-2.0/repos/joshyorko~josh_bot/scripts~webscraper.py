import pprint
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
import sys


class WebScraper:
    def __init__(self, model_name="gpt-3.5-turbo-0613", ollama_model=None):
        if ollama_model:
            self.llm = ChatOllama(
                model=ollama_model,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        else:
            self.llm = ChatOpenAI(model=model_name)

    #def extract(self, content: str, schema: dict):
    #    return create_extraction_chain(schema=schema, llm=self.llm).run(content)
#
    def scrape_with_playwright(self,urls, schema):
        loader = AsyncChromiumLoader(urls)
        docs = loader.load()
   
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, tags_to_extract=["span"]
        )
        #print(docs_transformed[0])
        print(f"Number of documents: {len(docs_transformed)}")
        
        # Grab the first 1000 tokens of the site
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=25
        )
        splits = splitter.split_documents(docs_transformed)
        vectorstore = Chroma.from_documents(persist_directory='testing_wsj',documents=splits, embedding=GPT4AllEmbeddings())

        # Process the first split
        #extracted_content = self.extract(schema=schema, content=splits[0].page_content)
        #pprint.pprint(extracted_content)
        return print('Vectorstore Created')

    def run_scraper(self, urls, schema):
        return self.scrape_with_playwright(urls, schema)

# Usage:
schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

# Usage with ChatOpenAI:
#web_scraper_openai = WebScraper()
#extracted_content_openai = web_scraper_openai.run_scraper(["https://www.wsj.com"], schema)

# Usage with ChatOllama:
web_scraper_ollama = WebScraper(ollama_model="mistral:7b")
extracted_content_ollama = web_scraper_ollama.run_scraper(["https://www.wsj.com"], schema)

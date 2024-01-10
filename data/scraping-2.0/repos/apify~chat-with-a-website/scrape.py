import os

from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain.document_loaders import ApifyDatasetLoader
from langchain.document_loaders.base import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables from a .env file
load_dotenv()

if __name__ == '__main__':
    apify_client = ApifyClient(os.environ.get('APIFY_API_TOKEN'))
    website_url = os.environ.get('WEBSITE_URL')
    print(f'Extracting data from "{website_url}". Please wait...')
    actor_run_info = apify_client.actor('apify/website-content-crawler').call(
        run_input={'startUrls': [{'url': website_url}]}
    )
    print('Saving data into the vector database. Please wait...')
    loader = ApifyDatasetLoader(
        dataset_id=actor_run_info['defaultDatasetId'],
        dataset_mapping_function=lambda item: Document(
            page_content=item['text'] or '', metadata={'source': item['url']}
        ),
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory='db2',
    )
    vectordb.persist()
    print('All done!')

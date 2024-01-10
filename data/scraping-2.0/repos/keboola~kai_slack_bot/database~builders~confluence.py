import os
import re
import openai
import logging
from dotenv import load_dotenv
from atlassian import Confluence
from bs4 import BeautifulSoup
import json
import pinecone
import time

from llama_index import GPTVectorStoreIndex, ServiceContext, Document, set_global_service_context
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext

from llama_index.llms import OpenAI

from llama_index.llm_predictor import LLMPredictor


from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)


openai.api_key=os.getenv('OPENAI_API_KEY')

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Confluence credentials
CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL')
CONFLUENCE_USERNAME = os.environ.get('CONFLUENCE_USERNAME')
CONFLUENCE_PASSWORD = os.environ.get('CONFLUENCE_PASSWORD')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Folder to save downloaded files
SAVE_FOLDER = 'downloaded_data'
LIMIT = 100 


def sanitize_filename(filename):
    # Replace slashes with underscores
    return re.sub(r'[/\\]', '_', filename)


def save_results(results, metadata, directory):
    for result in results:
        content_filename = os.path.join(directory, sanitize_filename(result['title']) + ".txt")
        metadata_filename = os.path.join(directory, sanitize_filename(result['title']) + ".json")

        html_content = result['body']['storage']['value']
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        text = result['title'] + '\n\n' + text

        with open(content_filename, 'w', encoding='utf-8') as file:
            file.write(text)

        with open(metadata_filename, 'w', encoding='utf-8') as file:
            json.dump(metadata, file)


def get_metadata(confluence, results):
    page_id = results[0].get("id")
    if page_id:
        data = confluence.get_page_by_id(page_id)

        page_metadata = {
            'id': data.get('id', ''),
            'CreatedDate': data['history'].get('createdDate', ''),
            'LastUpdatedDate': data['version'].get('when', ''),
            'Title': data.get('title', ''),
            'Creator': data['history']['createdBy'].get('displayName', ''),
            'LastModifier': data['version']['by'].get('displayName', ''),
            'url': data['_links'].get('base', '') + '/pages/' + data.get('id', ''),
            'Space': data['space'].get('name', '')
        }

        return page_metadata
    return None


def download_confluence_pages(confluence, save_folder: str = SAVE_FOLDER):
    spaces = confluence.get_all_spaces()
    for space in spaces.get("results"):
        logging.info(f"Downloading Confluence space: {space['name']}")

        content = confluence.get_space_content(space['key'])
        while True:
            subdir = os.path.join(save_folder, space['name'])
            os.makedirs(subdir, exist_ok=True)

            page = content.get("page")
            results = page.get("results")
            size = page.get("size")

            if not results:
                logging.info(f"No results for {space['name']}")
                break

            metadata = get_metadata(confluence, results)

            save_results(results, metadata, subdir)

            if size == LIMIT:
                start = page.get("start") + LIMIT
                content = confluence.get_space_content(space['key'], start=start, limit=LIMIT)
                page = content.get("page")
                results = page.get("results")
                metadata = get_metadata(confluence, results)
                save_results(results, metadata, subdir)
            else:
                break


def read_file_as_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def get_file_metadata(file_path) -> dict:
    metadata_path = file_path.replace(".txt", ".json")
    md = read_file_as_string(metadata_path)
    md = json.loads(md)
    if md:
        return md
    return {}


def main():
    confluence = Confluence(url=CONFLUENCE_URL, username=CONFLUENCE_USERNAME, password=CONFLUENCE_PASSWORD)

    # Download attachments from Confluence
    download_confluence_pages(confluence,
                              save_folder="database/datadir/confluence")

    print('Data download complete!')


def store_index():

    doc_titles = []
    doc_paths = []

    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)

    llm_predictor = LLMPredictor(llm=llm)

    metadata_extractor = MetadataExtractor(
    extractors=[
        #TitleExtractor(nodes=2),
        QuestionsAnsweredExtractor(questions=3, llm_predictor=llm_predictor),
        #SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=5, llm_predictor=llm_predictor),
    ],
    )

    node_parser = SimpleNodeParser(text_splitter=text_splitter, metadata_extractor=metadata_extractor)
      

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    set_global_service_context(service_context)

    for dirpath, dirnames, filenames in os.walk("database/datadir/confluence"):
        for filename in filenames:
            if filename.endswith(".txt"):
                subdir_name = os.path.basename(dirpath)
                file_name = os.path.splitext(filename)[0]

                doc_titles.append(subdir_name + " - " + file_name)
                doc_paths.append(os.path.join(dirpath, filename))

    docs = []
    for title, path in zip(doc_titles, doc_paths):

        if str(path).endswith(".txt"):


            text = read_file_as_string(path)
            extra_info = get_file_metadata(path)


            docs.append(Document(
                text=text,
                doc_id=title,
                extra_info=extra_info
            ))
            print('Document added: ' + title)

    print('Documents added: ' + str(len(docs)))

    start = time.time()
    print(time.time())
    

    nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)
 

    print(time.time() - start) 
    print('Nodes added: ' + str(len(nodes)))
          
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_ENVIRONMENT']
    )

    # create the index if it does not exist already
    index_name = 'kaidev'

    pinecone_index = pinecone.Index(index_name)

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    GPTVectorStoreIndex(
        nodes, storage_context=storage_context,
        service_context=service_context,
        show_progress=True
    )

if __name__ == "__main__":
    #main()
    store_index()

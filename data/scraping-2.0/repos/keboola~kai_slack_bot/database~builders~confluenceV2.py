import os
import pickle
import re
import logging
import json
import time
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from atlassian import Confluence
import pinecone
from datetime import datetime


from llama_index import (
    GPTVectorStoreIndex,
    ServiceContext,
    Document,
    set_global_service_context,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    # TitleExtractor,
    QuestionsAnsweredExtractor,
    # SummaryExtractor,
    KeywordExtractor,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

openai.api_key=os.getenv('OPENAI_API_KEY')


class ConfluenceDataExtractor:
    def __init__(self, confluence_url, confluence_username, confluence_password, save_folder):
        self.confluence = Confluence(
            url=confluence_url, username=confluence_username, password=confluence_password
        )
        self.save_folder = save_folder

    def sanitize_filename(self, filename):
        return re.sub(r"[/\\]", "_", filename)

    def save_results(self, results, metadata, directory):
        for result in results:
            content_filename = os.path.join(
                directory, self.sanitize_filename(result["title"]) + ".txt"
            )
            metadata_filename = os.path.join(
                directory, self.sanitize_filename(result["title"]) + ".json"
            )

            html_content = result["body"]["storage"]["value"]
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            text = result["title"] + "\n\n" + text

            with open(content_filename, "w", encoding="utf-8") as file:
                file.write(text)

            with open(metadata_filename, "w", encoding="utf-8") as file:
                json.dump(metadata, file)

    def get_metadata(self, results):
        page_id = results[0].get("id")
        if page_id:
            data = self.confluence.get_page_by_id(page_id)

            page_metadata = {
                "id": data.get("id", ""),
                "CreatedDate": data["history"].get("createdDate", ""),
                "LastUpdatedDate": data["version"].get("when", ""),
                "Title": data.get("title", ""),
                "Creator": data["history"]["createdBy"].get("displayName", ""),
                "LastModifier": data["version"]["by"].get("displayName", ""),
                "url": data["_links"].get("base", "") + "/pages/" + data.get("id", ""),
                "Space": data["space"].get("name", ""),
            }

            return page_metadata
        return {}

    def download_confluence_pages(self, limit=100):
        spaces = self.confluence.get_all_spaces()
        for space in spaces.get("results"):
            logging.info(f"Downloading Confluence space: {space['name']}")

            content = self.confluence.get_space_content(space["key"])
            while True:
                subdir = os.path.join(self.save_folder, space["name"])
                os.makedirs(subdir, exist_ok=True)

                page = content.get("page")
                results = page.get("results")
                size = page.get("size")

                if not results:
                    logging.info(f"No results for {space['name']}")
                    break

                metadata = self.get_metadata(results)

                # Check if the document is already downloaded and up-to-date
                for result in results:
                    metadata_filename = os.path.join(
                        subdir, self.sanitize_filename(result["title"]) + ".json"
                    )

                    if os.path.exists(metadata_filename):
                        with open(metadata_filename, "r", encoding="utf-8") as file:
                            existing_metadata = json.load(file)
                            if (
                                 metadata["LastUpdatedDate"]
                                == existing_metadata.get("LastUpdatedDate")
                            ):
                                logging.info(
                                    f"Document '{result['title']}' is up-to-date. Skipping download."
                                )
                                continue

                self.save_results(results, metadata, subdir)

                if size == limit:
                    start = page.get("start") + limit
                    content = self.confluence.get_space_content(
                        space["key"], start=start, limit=limit
                    )
                    page = content.get("page")
                    results = page.get("results")
                    metadata = self.get_metadata(results)

                    # Check if the document is already downloaded and up-to-date
                    for result in results:
                        metadata_filename = os.path.join(
                            subdir, self.sanitize_filename(result["title"]) + ".json"
                        )

                        if os.path.exists(metadata_filename):
                            with open(metadata_filename, "r", encoding="utf-8") as file:
                                existing_metadata = json.load(file)
                                if (
                                    metadata["LastUpdatedDate"]
                                    == existing_metadata.get("LastUpdatedDate")
                                ):
                                    logging.info(
                                        f"Document '{result['title']}' is up-to-date. Skipping download."
                                    )
                                    continue

                    self.save_results(results, metadata, subdir)
                else:
                    break



class DocumentIndexCreator:
    def __init__(self, save_folder, index_filename, batch_size=100):
        self.save_folder = save_folder
        self.index_filename = index_filename
        self.batch_size = batch_size
        self.doc_titles = []
        self.doc_paths = []
        self.doc_ids = []
        self.doc_embeddings = {}
        self.nodes_embeddings = {}  # Separate dictionary to store node embeddings

        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        self.llm_predictor = LLMPredictor(llm=self.llm)
        self.text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
        self.metadata_extractor = MetadataExtractor(
            extractors=[
                # TitleExtractor(nodes=2),
                QuestionsAnsweredExtractor(
                    questions=3, llm_predictor=self.llm_predictor
                ),
                # SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=5, llm_predictor=self.llm_predictor),
            ]
        )
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter, metadata_extractor=self.metadata_extractor
        )
        self.last_runtime = self.load_last_runtimes()

        self.load_documents()
    
    def sanitize_filename(self, filename):
        return re.sub(r"[/\\]", "_", filename)
    
    def read_file_as_string(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def get_file_metadata(self, file_path):
        metadata_path = file_path.replace(".txt", ".json")
        md = self.read_file_as_string(metadata_path)
        md = json.loads(md)
        if md:
            return md
        return {}

    def load_last_runtimes(self):
        if os.path.exists("runtimes.json"):
            with open("runtimes.json", "r") as file:
                return json.load(file).get("LastRuntime", {})
        return {}


    def update_last_runtime(self, doc_path):
        self.last_runtime[doc_path] = time.time() # Update the specific document's last runtime
        self.save_last_runtimes()

    def save_last_runtimes(self):
        with open("runtimes.json", "w") as file:
            json.dump({"LastRuntime": self.last_runtime}, file) # Sa

    def load_documents(self):
        for dirpath, dirnames, filenames in os.walk(self.save_folder):
            for filename in filenames:
                if filename.endswith(".txt"):
                    subdir_name = os.path.basename(dirpath)
                    file_name = os.path.splitext(filename)[0]

                    doc_path = os.path.join(dirpath, filename)

                    metadata_path = os.path.join(
                        dirpath, self.sanitize_filename(file_name) + ".json"
                    )
                    metadata = self.get_file_metadata(metadata_path)

                    last_updated_date_str = metadata.get("LastUpdatedDate", "")
                    if last_updated_date_str:
                        last_updated_date = datetime.fromisoformat(last_updated_date_str[:-1])
                        last_runtime = datetime.fromtimestamp(self.last_runtime.get(doc_path, 0)) # Use the specific runtime
                        if last_updated_date > last_runtime:
                            self.doc_titles.append(subdir_name + " - " + file_name)
                            self.doc_paths.append(doc_path)



    def index_documents(self):
        nodes = []
        for title, path in zip(self.doc_titles, self.doc_paths):
            if path.endswith(".txt"):
                text = self.read_file_as_string(path)
                extra_info = self.get_file_metadata(path)

                nodes.append(Document(text=text, doc_id=title, extra_info=extra_info))
                print("Document added: " + title)

                if len(nodes) >= self.batch_size:
                    self.process_batch(nodes)
                    nodes = []

        if nodes:
            self.process_batch(nodes)

    def process_batch(self, nodes):
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embed_model, node_parser=self.node_parser
        )
        set_global_service_context(service_context)

        start = time.time()
        print(time.time())
        
        parsed_nodes = self.node_parser.get_nodes_from_documents(nodes, show_progress=True)

        print(time.time() - start)
        print("Nodes added: " + str(len(parsed_nodes)))
        for node in parsed_nodes:
            doc_path = node.ref_doc_id # Assuming ref_doc_id contains the document path
            self.update_last_runtime(doc_path)
      
        self.update_index(parsed_nodes)
   


      
        self.save_index()
    def update_index(self, nodes):
        for node in nodes:
            if node.ref_doc_id not in self.doc_ids:
                self.doc_ids.append(node.ref_doc_id)
                self.doc_embeddings[node.ref_doc_id] = node.embedding
            else:
                self.doc_embeddings[node.ref_doc_id] = node.embedding

            # Store node embeddings in a separate dictionary
            self.nodes_embeddings[node.ref_doc_id] = node.embedding

        self.save_index()

    def save_index(self):
        with open(self.index_filename, "wb") as file:
            index_data = {
                "doc_embeddings": self.doc_embeddings,
                "nodes_embeddings": self.nodes_embeddings,
            }
            pickle.dump(index_data, file)

    def load_index(self):
        if os.path.exists(self.index_filename):
            with open(self.index_filename, "rb") as file:
                index_data = pickle.load(file)
                self.doc_embeddings = index_data.get("doc_embeddings", {})
                self.nodes_embeddings = index_data.get("nodes_embeddings", {})


def create_and_load_index(index_name, nodes):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    GPTVectorStoreIndex(
        nodes,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )


if __name__ == "__main__":
    # Load environment variables from .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

    # Constants
    SAVE_FOLDER = "downloaded_data"
    INDEX_NAME = "kaidev"
    INDEX_FILENAME = "index_data.pkl"
    BATCH_SIZE = 1

    # Create objects and perform tasks
    downloader = ConfluenceDataExtractor(
        confluence_url=os.environ.get("CONFLUENCE_URL"),
        confluence_username=os.environ.get("CONFLUENCE_USERNAME"),
        confluence_password=os.environ.get("CONFLUENCE_PASSWORD"),
        save_folder=SAVE_FOLDER,
    )
    downloader.download_confluence_pages()

    indexer = DocumentIndexCreator(
        save_folder=SAVE_FOLDER, index_filename=INDEX_FILENAME, batch_size=BATCH_SIZE
    )
    indexer.load_documents()
    indexer.index_documents()

    create_and_load_index(index_name=INDEX_NAME, nodes=indexer.doc_ids, )  # Only push document IDs to Pinecone
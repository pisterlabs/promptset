#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gpt4all
import psutil
import json
import time
import sys
import getopt
import os
import glob
import hashlib
import chromadb
import logging
import nltk
from chromadb.config import Settings
from chromadb.api.segment import API
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

try:
    nltk.data.find('punkt')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

class MyEmlLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


class ChatBot:
    def __init__(self):
        self.total_memory = psutil.virtual_memory()[0]
        self.used_memory = psutil.virtual_memory()[3]
        self.free_memory = psutil.virtual_memory()[1]
        self.used_percent = psutil.virtual_memory()[2]
        self.script_path = os.path.normpath(os.path.dirname(__file__))
        self.module_path = os.path.normpath(os.path.dirname(self.script_path.rstrip("/")))
        self.persist_directory = (f'{self.module_path}'
                                  f'/chromadb')
        self.model_directory = (f'{self.module_path}'
                                f'/models')
        self.source_directory = self.module_path
        self.openai_api_base = ""
        self.openai_api_key = False
        self.model = "llama-2-7b-chat.ggmlv3.q4_0.bin" #"nous-hermes-13b.ggmlv3.q4_0.bin"
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))
        self.model_engine = "GPT4All"
        self.embeddings_model_name = "all-mpnet-base-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        self.chunk_overlap = 69
        self.chunk_size = 639
        self.target_source_chunks = 6
        self.mute_stream = True
        self.hide_source = False
        self.model_n_ctx = 2127
        self.model_n_batch = 9
        self.bytes = 1073741824
        self.collection = None
        self.collection_name = "genius"
        self.vectorstore = "chromadb"
        self.chromadb_client = None
        self.pgvector_user = ""
        self.pgvector_password = ""
        self.pgvector_host = "localhost"
        self.pgvector_port = "5432"
        self.pgvector_driver = "psycopg2"
        self.pgvector_database = ""
        self.chroma_settings = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        self.loader_mapping = {
            ".csv": (CSVLoader, {}),
            ".docx": (Docx2txtLoader, {}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".enex": (EverNoteLoader, {}),
            ".eml": (MyEmlLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PyMuPDFLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            # Add more mappings for other file extensions and loaders as needed
        }
        self.payload = None

    def set_chromadb_directory(self, directory):
        self.persist_directory = f'{directory}/chromadb'
        if not os.path.isdir(self.persist_directory):
            logging.info(f"Making chromadb directory: {self.persist_directory}")
            os.mkdir(self.persist_directory)
        self.chroma_settings = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )

    def set_models_directory(self, directory):
        self.model_directory = f'{directory}'
        if not os.path.isdir(self.model_directory):
            logging.info(f"Making models directory: {self.model_directory}")
            os.mkdir(self.model_directory)
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))

    def check_hardware(self):
        self.total_memory = psutil.virtual_memory()[0]
        self.used_memory = psutil.virtual_memory()[3]
        self.free_memory = psutil.virtual_memory()[1]
        self.used_percent = psutil.virtual_memory()[2]
        logging.info(f'RAM Utilization: {round(self.used_percent, 2)}%\n'
                     f'\tUsed  RAM: {round(float(self.used_memory / self.bytes), 2)} GB\n'
                     f'\tFree  RAM: {round(float(self.free_memory / self.bytes), 2)} GB\n'
                     f'\tTotal RAM: {round(float(self.total_memory / self.bytes), 2)} GB\n\n')

    def chat(self, prompt: str) -> dict:
        db = None
        if self.vectorstore == "chromadb":
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings,
                        client_settings=self.chroma_settings, client=self.chromadb_client)
        elif self.vectorstore == "pgvector":
            db = PGVector(
                collection_name=self.collection_name,
                connection_string=PGVector.connection_string_from_db_params(
                    driver=self.pgvector_driver,
                    host=self.pgvector_host,
                    port=int(self.pgvector_port),
                    database=self.pgvector_database,
                    user=self.pgvector_user,
                    password=self.pgvector_password,
                ),
                embedding_function=self.embeddings,
            )
        retriever = db.as_retriever(search_kwargs={"k": self.target_source_chunks})
        callbacks = [] if self.mute_stream else [StreamingStdOutCallbackHandler()]
        # Download model
        self.set_models_directory(directory=self.model_directory)
        if os.path.isfile(os.path.join(self.model_directory, self.model)):
            print(f'Already downloaded model: {self.model_path}')
        else:
            print(f'Model was not found, downloading...')
            gpt4all.GPT4All.retrieve_model(self.model, self.model_directory, allow_download=True)
        # Prepare the LLM
        match self.model_engine.lower():
            case "llamaccp":
                llm = LlamaCpp(model=self.model_path, max_tokens=self.model_n_ctx, n_batch=self.model_n_batch,
                               callbacks=callbacks, verbose=False)
            case "gpt4all":
                llm = GPT4All(model=self.model_path, max_tokens=self.model_n_ctx, backend='gptj',
                              n_batch=self.model_n_batch, callbacks=callbacks, verbose=True)
            case "openai":
                llm = OpenAI(temperature=0.9)
            case _default:
                # raise exception if model_type is not supported
                raise Exception(f"Model type {self.model_engine} is not supported. "
                                f"Please choose one of the following: LlamaCpp, GPT4All")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                         return_source_documents=not self.hide_source)
        start = time.time()
        res = qa(prompt)
        answer, docs = res['result'], [] if self.hide_source else res['source_documents']
        end = time.time()
        documents = ""
        for document in docs:
            documents = f'{documents}\n{document.page_content}\n{document.metadata["source"]}'
        self.payload = {
            'model': self.model_engine,
            'embeddings_model': self.embeddings_model_name,
            'prompt': prompt,
            'answer': answer,
            'start_time': start,
            'end_time': end,
            'time_to_respond': round(end - start, 2),
            'batch_token': self.model_n_batch,
            'max_token_limit': self.model_n_ctx,
            'chunks': self.target_source_chunks,
            'sources': documents
        }
        return self.payload

    def assimilate(self):
        if self.vectorstore == "chromadb":
            chromadb_client = chromadb.PersistentClient(settings=self.chroma_settings , path=self.persist_directory)
            if self.does_vectorstore_exist():
                # Update and store locally vectorstore
                print(f"Appending to existing vectorstore at {self.persist_directory}")
                db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings,
                            client_settings=self.chroma_settings, client=chromadb_client)
                collection = db.get()
                documents = self.process_documents([metadata['source'] for metadata in collection['metadatas']])
                print(f"Creating embeddings. This may take a few minutes...")
                for batched_chromadb_insertion in self.batch_chromadb_insertions(chromadb_client, documents):
                    db.add_documents(batched_chromadb_insertion)
            else:
                # Create and store locally vectorstore
                print("Creating new vectorstore")
                documents = self.process_documents()
                print(f"Creating embeddings. This may take a few minutes...")
                # Create the db with the first batch of documents to insert
                batched_chromadb_insertions = self.batch_chromadb_insertions(chromadb_client, documents)
                first_insertion = next(batched_chromadb_insertions)
                db = Chroma.from_documents(first_insertion, self.embeddings, persist_directory=self.persist_directory,
                                           client_settings=self.chroma_settings, client=chromadb_client)
                # Add the rest of batches of documents
                for batched_chromadb_insertion in batched_chromadb_insertions:
                    db.add_documents(batched_chromadb_insertion)
        elif self.vectorstore == "pgvector":
            if self.does_vectorstore_exist():
                documents = self.process_documents()
                db = PGVector(
                    collection_name=self.collection_name,
                    connection_string=PGVector.connection_string_from_db_params(
                        driver=self.pgvector_driver,
                        host=self.pgvector_host,
                        port=int(self.pgvector_port),
                        database=self.pgvector_database,
                        user=self.pgvector_user,
                        password=self.pgvector_password,
                    ),
                    embedding_function=self.embeddings,
                )
                db.add_documents(documents)
            else:
                documents = self.process_documents()
                PGVector.from_documents(
                    embedding=self.embeddings,
                    documents=documents,
                    collection_name=self.collection_name,
                    connection_string=PGVector.connection_string_from_db_params(
                        driver=self.pgvector_driver,
                        host=self.pgvector_host,
                        port=int(self.pgvector_port),
                        database=self.pgvector_database,
                        user=self.pgvector_user,
                        password=self.pgvector_password,
                    ),
                )

    def load_single_document(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        if ext in self.loader_mapping:
            loader_class, loader_args = self.loader_mapping[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_dir: str, ignored_files: List[str] = []):
        all_files = []
        for ext in self.loader_mapping:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
            )
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
        with Pool(processes=os.cpu_count()) as pool:
            documents = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self.load_single_document, filtered_files)):
                    documents.extend(docs)
                    pbar.update()

        return documents

    def generate_md5_checksum(self, file):
        with open(file, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5_checksum = hashlib.md5(data).hexdigest()
        return md5_checksum

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        print(f"Loading documents from {self.source_directory}")
        documents = self.load_documents(self.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            return []
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(documents)
        print(f"Split into {len(documents)} chunks of text (max. {self.chunk_size} tokens each)")
        return documents

    def batch_chromadb_insertions(self, chromadb_client: API, documents: List[Document]) -> List[Document]:
        # Get max batch size.
        max_batch_size = chromadb_client.max_batch_size
        for i in range(0, len(documents), max_batch_size):
            yield documents[i:i + max_batch_size]

    def does_vectorstore_exist(self) -> bool:
        if self.vectorstore == "chromadb":
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            if not db.get()['documents']:
                return False
            return True
        elif self.vectorstore == "pgvector":
            db = PGVector(
                collection_name=self.collection_name,
                connection_string=PGVector.connection_string_from_db_params(
                    driver=self.pgvector_driver,
                    host=self.pgvector_host,
                    port=int(self.pgvector_port),
                    database=self.pgvector_database,
                    user=self.pgvector_user,
                    password=self.pgvector_password,
                ),
                embedding_function=self.embeddings,
            )
            if not db:
                return False
            return True



def usage():
    print(f'Usage:\n'
          f'-h | --help               [ See usage for script ]\n'
          f'-a | --assimilate         [ Assimilate knowledge from media provided in directory ]\n'
          f'   | --batch-token        [ Number of tokens per batch ]\n'
          f'   | --chromadb-directory [ Directory for chromadb persistent storage ]\n'
          f'   | --chunks             [ Number of chunks to use ]\n'
          f'-e | --embeddings-model   [ Embeddings model to use https://www.sbert.net/docs/pretrained_models.html ]\n'
          f'   | --hide-source        [ Hide source of answer ]\n'
          f'-j | --json               [ Export to JSON ]\n'
          f'   | --openai-token       [ OpenAI token ]\n'
          f'   | --openai-api         [ OpenAI API Url ]\n'
          f'   | --pgvector-user      [ PGVector user ]\n'
          f'   | --pgvector-password  [ PGVector password ]\n'
          f'   | --pgvector-host      [ PGVector host ]\n'
          f'   | --pgvector-port      [ PGVector port ]\n'
          f'   | --pgvector-database  [ PGVector database ]\n'
          f'   | --pgvector-driver    [ PGVector driver ]\n'
          f'-p | --prompt             [ Prompt for chatbot ]\n'
          f'   | --mute-stream        [ Mute stream of generation ]\n'
          f'-m | --model              [ Model to use from GPT4All https://gpt4all.io/index.html ]\n'          
          f'   | --max-token-limit    [ Maximum token to generate ]\n'
          f'   | --model-directory    [ Directory to store models ]\n'
          f'   | --model-engine       [ GPT4All, LlamaCPP, or OpenAI ]\n'
          f'\nExample:\n'
          f'genius-chatbot --assimilate "/directory/of/documents"\n'
          f'\n'
          f'genius-chatbot --prompt "What is the 10th digit of Pi?"\n'
          f'\n'
          f'genius-chatbot --prompt "Chatbots are cool because they" \\\n'
          f'\t--model "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin" \\\n'
          f'\t--model-engine "GPT4All" \\\n'
          f'\t--assimilate "/directory/of/documents" \\\n'
          f'\t--json\n')


def genius_chatbot(argv):
    geniusbot_chat = ChatBot()
    run_flag = False
    assimilate_flag = False
    json_export_flag = False
    prompt = 'Geniusbot is the smartest chatbot in existence.'
    try:
        opts, args = getopt.getopt(argv, 'a:he:jm:p:',
                                   ['help', 'assimilate=', 'prompt=', 'json',
                                    'batch-token=', 'chunks=', 'max-token-limit=',
                                    'hide-source', 'mute-stream',
                                    'embeddings-model=', 'model=', 'model-engine=',
                                    'model-directory=', 'chromadb-directory=',
                                    'openai-token=', 'openai-api=',
                                    'pgvector-user=','pgvector-password=','pgvector-host=','pgvector-port=',
                                    'pgvector-driver=','pgvector-database='])
    except getopt.GetoptError as e:
        usage()
        logging.error("Error: {e}\nExiting...")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-a', '--assimilate'):
            if os.path.exists(arg):
                geniusbot_chat.source_directory = str(arg)
                assimilate_flag = True
            else:
                logging.error(f'Path does not exist: {arg}')
                sys.exit(1)
        elif opt == '--batch-token':
            geniusbot_chat.model_n_batch = int(arg)
        elif opt == '--chunks':
            geniusbot_chat.target_source_chunks = int(arg)
        elif opt == '--chromadb-directory':
            geniusbot_chat.set_chromadb_directory(directory=str(arg))
            geniusbot_chat.vectorstore = "chromadb"
        elif opt in ('-j', '--json'):
            geniusbot_chat.json_export_flag = True
            geniusbot_chat.hide_source_flag = True
            geniusbot_chat.mute_stream_flag = True
        elif opt in ('-e', '--embeddings-model'):
            geniusbot_chat.embeddings_model_name = arg
            geniusbot_chat.embeddings = HuggingFaceEmbeddings(model_name=geniusbot_chat.embeddings_model_name)
        elif opt in ('-m', '--model'):
            geniusbot_chat.model = arg
            geniusbot_chat.model_path = os.path.normpath(
                os.path.join(geniusbot_chat.model_directory, geniusbot_chat.model))
            print(f"Model: {geniusbot_chat.model}")
        elif opt == '--openai-token':
            os.environ["OPENAI_API_KEY"] = arg
            geniusbot_chat.openai_api_key = True
        elif opt == '--openai-api':
            os.environ["OPENAI_API_BASE"] = arg
            geniusbot_chat.openai_api_base = arg
        elif opt == '--model-engine':
            geniusbot_chat.model_engine = arg
            if (geniusbot_chat.model_engine.lower() != "llamacpp"
                    and geniusbot_chat.model_engine.lower() != "gpt4all"
                    and geniusbot_chat.model_engine.lower() != "openai"):
                logging.error("model type not supported")
                usage()
                sys.exit(2)
        elif opt == '--model-directory':
            geniusbot_chat.set_models_directory(directory=str(arg))
        elif opt in ('-p', '--prompt'):
            prompt = str(arg)
            run_flag = True
        elif opt == '--hide-source':
            geniusbot_chat.hide_source_flag = True
        elif opt == '--pgvector-user':
            geniusbot_chat.pgvector_user = arg
        elif opt == '--pgvector-password':
            geniusbot_chat.pgvector_password = arg
        elif opt == '--pgvector-host':
            geniusbot_chat.pgvector_host = arg
            geniusbot_chat.vectorstore = "pgvector"
        elif opt == '--pgvector-port':
            geniusbot_chat.pgvector_port = arg
        elif opt == '--pgvector-driver':
            geniusbot_chat.pgvector_driver = arg
        elif opt == '--pgvector-database':
            geniusbot_chat.pgvector_database = arg
        elif opt == '--max-token-limit':
            geniusbot_chat.model_n_ctx = int(arg)
        elif opt == '--mute-stream':
            geniusbot_chat.mute_stream_flag = True

    if assimilate_flag:
        geniusbot_chat.assimilate()

    if geniusbot_chat.openai_api_key:
        geniusbot_chat.embeddings = OpenAIEmbeddings()

    if run_flag:
        if not geniusbot_chat.does_vectorstore_exist():
            geniusbot_chat.assimilate()
        if geniusbot_chat.vectorstore == "chromadb":
            geniusbot_chat.chromadb_client = chromadb.PersistentClient(settings=geniusbot_chat.chroma_settings,
                                                                       path=geniusbot_chat.persist_directory)
        logging.info('RAM Utilization Before Loading Model')
        geniusbot_chat.check_hardware()
        response = geniusbot_chat.chat(prompt=prompt)
        if json_export_flag:
            print(json.dumps(response, indent=4))
        else:
            print(f"\n\nQuestion: {response['prompt']}\n"
                  f"Answer: {response['answer']}\n"
                  f"Sources: {response['sources']}\n\n")
            logging.info('RAM Utilization After Loading Model')
        geniusbot_chat.check_hardware()


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    genius_chatbot(sys.argv[1:])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    genius_chatbot(sys.argv[1:])

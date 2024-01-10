import logging
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Tuple
from urllib.parse import unquote

from deep_translator import GoogleTranslator
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.docstore.wikipedia import Wikipedia
from langchain.tools import Tool, DuckDuckGoSearchRun
from pydantic import BaseModel, root_validator, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from scrapalot_main import get_llm_instance
from scripts import app_logs
from scripts.app_environment import chromaDB_manager, api_host, api_port, api_scheme
from scripts.app_qa_builder import process_database_question, process_query

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

app = FastAPI(title="scrapalot-chat API")

origins = [
    "http://localhost:3000", "http://localhost:8000", "https://scrapalot.com"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="scrapalot-research-assistant-ui/static"), name="static")

load_dotenv()


###############################################################################
# model classes
###############################################################################

class QueryBodyFilter(BaseModel):
    filter_document: bool = Field(False, description="Whether to filter the document or not.")
    filter_document_name: Optional[str] = Field(None, description="Name of the document to filter.")
    translate_chunks: bool = Field(True, description="Whether to translate chunks or not.")

    @root_validator(pre=True)
    def check_filter(cls, values):
        filter_document = values.get('filter_document')
        filter_document_name = values.get('filter_document_name')
        if filter_document and not filter_document_name:
            raise ValueError("filter_document is True but filter_document_name is not provided.")
        return values


class QueryLLMBody(BaseModel):
    database_name: str
    collection_name: str
    question: str
    locale: str
    filter_options: QueryBodyFilter


class QueryWeb(BaseModel):
    question: str
    locale: str
    filter_options: QueryBodyFilter


class TranslationBody(BaseModel):
    locale: str


class SourceDirectoryDatabase(BaseModel):
    name: str
    path: str


class SourceDirectoryFile(BaseModel):
    id: str
    name: str


class SourceDirectoryFilePaginated(BaseModel):
    total_count: int
    items: List[SourceDirectoryFile]


class TranslationItem(BaseModel):
    src_lang: str
    dst_lang: str
    text: str


class LLM:
    def __init__(self):
        self.instance = None

    def get_instance(self):
        if not self.instance:
            self.instance = get_llm_instance(StreamingStdOutCallbackHandler())
        return self.instance


class WebSearch:
    def __init__(self):
        self.tools = None
        self.react = None
        self.search = None

    def initialize(self, llm):
        doc_store = DocstoreExplorer(Wikipedia())
        duckduckgo_search = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Search",
                func=doc_store.search,
                description="Search for a term in the doc store.",
            ),
            Tool(
                name="Lookup",
                func=doc_store.lookup,
                description="Lookup a term in the doc store.",
            ),
            Tool(
                name='DuckDuckGo Search',
                func=duckduckgo_search.run,
                description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
            ),
        ]

        self.react = initialize_agent(self.tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        self.search = AgentExecutor.from_agent_and_tools(
            agent=self.react.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            early_stopping_method="generate",
        )

    def get_tools(self):
        return self.tools

    def get_web_search(self):
        return self.search  # returning search, which is an instance of AgentExecutor


web_search = WebSearch()
###############################################################################
# init
###############################################################################
chat_history = []
llm_manager = LLM()
executor = ThreadPoolExecutor(max_workers=5)


@app.on_event("startup")
async def startup_event():
    llm = llm_manager.get_instance()
    web_search.initialize(llm)
    app_logs.initialize_logging()


def get_tools():
    return web_search.get_tools()


def get_agent():
    return web_search.get_web_search()


###############################################################################
# helper functions
###############################################################################

def get_llm():
    return llm_manager.get_instance()


def list_of_collections(database_name: str):
    client = chromaDB_manager.get_client(database_name)
    return client.list_collections()


def create_database(database_name):
    directory_path = os.path.join(".", "source_documents", database_name)
    db_path = f"./db/{database_name}"
    os.makedirs(directory_path)
    os.makedirs(db_path)
    set_key('.env', 'INGEST_SOURCE_DIRECTORY', directory_path)
    set_key('.env', 'INGEST_PERSIST_DIRECTORY', db_path)
    logging.debug(f"Created new database: {directory_path}")
    return directory_path, db_path


async def get_files_from_dir(database: str, page: int, items_per_page: int) -> Tuple[List[SourceDirectoryFile], int]:
    all_files = []

    for r, dirs, files in os.walk(database):
        for file in sorted(files, reverse=True):  # Sort files in descending order.
            if not file.startswith('.'):
                all_files.append(SourceDirectoryFile(id=str(uuid.uuid4()), name=file))
    start = (page - 1) * items_per_page
    end = start + items_per_page
    return all_files[start:end], len(all_files)


def run_ingest(database_name: str, collection_name: Optional[str] = None):
    if database_name and not collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name], check=True)
    if database_name and collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name, "--collection", collection_name], check=True)


async def get_database_file_response(absolute_file_path: str) -> Union[FileResponse]:
    return FileResponse(absolute_file_path)


###############################################################################
# API
###############################################################################
@app.get("/api")
async def root():
    return {"ping": "pong!"}


@app.get('/api/databases')
async def get_database_names_and_collections():
    base_dir = "./db"
    try:
        database_names = \
            sorted([name for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))])

        database_info = []
        for database_name in database_names:
            collections = list_of_collections(database_name)
            database_info.append({
                'database_name': database_name,
                'collections': collections
            })

        return database_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/database/{database_name}/new")
async def create_new_database(database_name: str):
    try:
        create_database(database_name)
        return {"message": "OK", "database_name": database_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/{database_name}", response_model=SourceDirectoryFilePaginated)
async def get_database_files(database_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    files, total_count = await get_files_from_dir(database_dir, page, items_per_page)
    return {"total_count": total_count, "items": files}


@app.get("/api/database/{database_name}/collection/{collection_name}", response_model=List[SourceDirectoryFile])
async def get_database_collection_files(database_name: str, collection_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    collection_dir = os.path.join(absolute_base_dir, database_name, collection_name)
    if not os.path.exists(collection_dir) or not os.path.isdir(collection_dir):
        raise HTTPException(status_code=404, detail="Collection not found")
    files = await get_files_from_dir(collection_dir, page, items_per_page)
    return files


@app.get("/api/database/{database_name}/file/{file_name}", response_model=None)
async def get_database_file(database_name: str, file_name: str) -> Union[HTMLResponse, FileResponse]:
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    absolute_file_path = os.path.join(database_dir, unquote(file_name))
    if not os.path.exists(absolute_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return await get_database_file_response(absolute_file_path)


@app.post('/api/query-llm')
async def query_files(body: QueryLLMBody, llm=Depends(get_llm)):
    database_name = body.database_name
    collection_name = body.collection_name
    question = body.question
    locale = body.locale
    filter_options = body.filter_options
    translate_chunks = filter_options.translate_chunks

    try:
        if locale != 'en':
            question = GoogleTranslator(source=locale, target='en').translate(question)

        qa = await process_database_question(database_name, llm, collection_name, filter_options.filter_document, filter_options.filter_document_name)

        answer, docs = process_query(qa, question, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

        if locale != 'en':
            answer = GoogleTranslator(source='en', target=locale).translate(answer)

        source_documents = []
        for doc in docs:
            if translate_chunks:
                doc.page_content = GoogleTranslator(source='en', target=locale).translate(doc.page_content)

            document_data = {
                'content': doc.page_content,
                'link': doc.metadata['source'],
            }
            if 'page' in doc.metadata:
                document_data['page'] = doc.metadata['page']
            if 'total_pages' in doc.metadata:
                document_data['total_pages'] = doc.metadata['total_pages']

            source_documents.append(document_data)

        response = {
            'answer': answer,
            'source_documents': source_documents
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/query-web')
async def query_web(body: QueryWeb, agent=Depends(get_agent)):
    question = body.question
    locale = body.locale
    filter_options = body.filter_options
    translate_chunks = filter_options.translate_chunks

    try:

        if locale != 'en':
            question = GoogleTranslator(source=locale, target='en').translate(question)

        result = agent({"input": question})
        observations = []
        for step in result["intermediate_steps"]:
            observations.append(step)

        source_documents = []
        for doc in observations:
            content = doc[1]
            link = doc[0].tool_input

            if translate_chunks:
                content = GoogleTranslator(source='en', target=locale).translate(content)
                link = GoogleTranslator(source='en', target=locale).translate(link)

            source_documents.append({"content": content, "link": link})

        answer = result["output"]
        if locale != 'en':
            answer = GoogleTranslator(source='en', target=locale).translate(answer)

        response = {
            'answer': answer,
            'source_documents': source_documents
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_files(request: Request):
    form = await request.form()
    database_name = form['database_name']
    collection_name = form.get('collection_name')  # Optional field

    files = form["files"]  # get files from form data

    # make sure files is a list
    if not isinstance(files, list):
        files = [files]

    saved_files = []
    source_documents = os.path.join(".", "source_documents")
    try:
        for file in files:
            content = await file.read()  # read file content
            if collection_name and database_name != collection_name:
                file_path = os.path.join(source_documents, database_name, collection_name, file.filename)
            else:
                file_path = os.path.join(source_documents, database_name, file.filename)

            saved_files.append(file_path)
            with open(file_path, "wb") as f:
                f.write(content)

            # assuming run_ingest is defined elsewhere
            run_ingest(database_name, collection_name)

            response = {
                'message': "OK",
                'files': saved_files,
                "database_name": database_name
            }
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def translate(item: TranslationItem):
    return {"translated_text": GoogleTranslator(source=item.src_lang, target=item.dst_lang).translate(item.text)}


###############################################################################
# Frontend
###############################################################################
@app.get("/")
def home():
    return FileResponse('scrapalot-research-assistant-ui/index.html')


@app.get("/{catch_all:path}")
def read_root(catch_all: str):
    logging.debug(f'Browsing through: {catch_all}')
    return FileResponse('scrapalot-research-assistant-ui/index.html')


# commented out, because we use web UI
if __name__ == "__main__":
    import uvicorn

    path = 'api'
    # cert_path = "cert/cert.pem"
    # key_path = "cert/key.pem"
    logging.debug(f"Scrapalot API is now available at {api_scheme}://{api_host}:{api_port}/{path}")
    uvicorn.run(app, host=api_host, port=int(api_port))
    # uvicorn.run(app, host=host, port=int(port), ssl_keyfile=key_path, ssl_certfile=cert_path)

from pathlib import Path

import dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PDFPlumberLoader,
)
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores import Weaviate

from ai_document_search_backend.utils.relative_path_from_file import (
    relative_path_from_file,
)

# Start Weaviate with `docker compose -f docker-compose-weaviate.yml up -d`
WEAVIATE_URL = "http://localhost:8080"
PDF_FILE_PATH = relative_path_from_file(__file__, "../../data/pdfs/NO0010914682_LA_20201217.pdf")
PDF_DIR_PATH = relative_path_from_file(__file__, "../../data/pdfs_subset/")

OPENAI_API_KEY = dotenv.dotenv_values()["APP_OPENAI_API_KEY"]

# 1.+2. Load + Split

# Using PyPDF
# Load PDF using pypdf into array of documents, where each document contains the page content
# and metadata with page number.
loader = PyPDFLoader(PDF_FILE_PATH)
data_pypdf = loader.load()

# PyPDF Directory
# Load PDFs from directory.
loader = PyPDFDirectoryLoader(PDF_DIR_PATH)
data_pypdf_dir = loader.load()

# Using Unstructured
loader = UnstructuredPDFLoader(PDF_FILE_PATH, mode="paged")
data = loader.load()  # only 44 pages, should be 45

# Using PyMuPDF
# This is the fastest of the PDF parsing options, and contains detailed metadata about the PDF and its pages,
# as well as returns one document per page.
loader = PyMuPDFLoader(PDF_FILE_PATH)
data_pymupdf = loader.load()

# Using PDFPlumber
# Like PyMuPDF, the output Documents contain detailed metadata about the PDF and its pages,
# and returns one document per page.
loader = PDFPlumberLoader(PDF_FILE_PATH)
data_pdfplumber = loader.load()

# 3. Store
vectorstore = Weaviate.from_documents(
    documents=data_pypdf,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    weaviate_url=WEAVIATE_URL,
    by_text=False,
)

# 4. Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"additional": ["certainty", "distance"]})

# 5.+6. Generate + Chat
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", output_key="answer", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, return_source_documents=True
)


def print_with_sources(result):
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print("Sources:")
    for source in result["source_documents"]:
        page = source.metadata["page"]
        name = Path(source.metadata["source"]).name
        additional = source.metadata["_additional"]
        certainty = round(additional["certainty"], 2)
        distance = round(additional["distance"], 2)
        print(f"\tPage {page} of {name} (certainty {certainty}, distance {distance})")
    print("-" * 50)


print_with_sources(qa("What is the Loan to value ratio?"))
print_with_sources(qa("How large is it?"))

import os
import openai
import langchain
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, wait_random_exponential
from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.schema import format_document
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableMap
from langchain.docstore.document import Document

import dotenv
from .az_ai_search_helper import *

env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
print(f"Loading environment from: {env_file_path}")
with open(env_file_path) as f:
    dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)

openai.api_type: str = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# openai_client = openai.AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=model,
    model=model,
    chunk_size=1,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_type="azure",
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

langchain.verbose = False

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 10

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}")


def get_pdf_text(files):
    text = ""
    for f in files:
        print(f"Processing {f.name}")
        # print(f"f object: {dir(f)}")
        reader = PdfReader(f)
        page_count = len(reader.pages)
        for page in reader.pages:
            text += page.extract_text()
        print(f"Page count: {page_count}")

    return text


def load_csv(csv_file):
    csv_loader = CSVLoader(file_path=csv_file)
    return csv_loader.load()


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def get_vectors(chunks):

    global embeddings
    if not embeddings:
        print("Embeddings not initialized. Initializing now.")
        return
    # text_embeddings = embeddings.embed_documents(chunks)
    # text_embeddings = [vectorize_with_delay(
    #     embeddings, chunk) for chunk in chunks]
    vector_store = FAISS.from_texts(chunks, embeddings)

    # text_embeddings = embed_with_delay(embeddings, chunks)
    # text_embedding_pairs = zip(chunks, text_embeddings)
    # text_embedding_pairs_list = list(text_embedding_pairs)
    # vector_store = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)

    return vector_store


def extract_and_split_documents(pdf_file):
    reader = PdfReader(pdf_file)
    docs = []
    page_num = 1
    for page in reader.pages:
        docs.append(Document(page_content=page.extract_text(),
                    metadata={'source': f"{pdf_file.name}", 'page': page_num}))
        page_num += 1

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    pages = text_splitter.split_documents(docs)

    return pages


def upload_docs_to_cogsearch_index(index_name, pdf_files):

    global embeddings
    if not embeddings:
        print("Embeddings not initialized. Initializing now.")
        return

    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    # Get Vector Store
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )

    if not vector_store:
        raise ValueError(
            f"Index {index_name} does not exist. Please create the index first.")

    for file in pdf_files:
        docs = extract_and_split_documents(file)
        print(f"Number of pages: {len(docs)}")
        vector_store.add_documents(documents=docs)


def get_az_search_vector_store(index_name):

    global embeddings
    if not embeddings:
        print("Embeddings not initialized. Initializing now.")
        return

    fields = get_index_fields(index_name, embeddings.embed_query)
    print(f"Index Fields: {fields}")

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_ADMIN_KEY"],
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=fields
    )

    return vector_store


@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10))
def vectorize_with_delay(embeddings, document):
    return embeddings.vectorize(document)


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def embed_with_delay(embeddings, document):
    return embeddings.embed_documents(document)


def get_conversation_chain(vector_store):

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_CHATGPT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_type="azure",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def get_chat_llm_chain(prompt, vector_store):
    chat_llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_CHATGPT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_type="azure",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0.5
    )
    retriever = vector_store.as_retriever()

    inputs = RunnableMap({
        "docs": RunnablePassthrough() | retriever,
        "question": RunnablePassthrough()
    })

    # Now we retrieve the documents
    context = RunnableMap({
        "context": RunnablePassthrough() | retriever | _combine_documents,
        "question": RunnablePassthrough(),
        "docs": RunnablePassthrough() | retriever,
    })

    context2 = RunnablePassthrough.assign(
        context=lambda x: _combine_documents(x["docs"]),
    )

    find_answer = RunnableMap({
        "answer":  prompt | chat_llm | StrOutputParser(),
        "docs": lambda x: x["docs"],
    })

    c2 = inputs | context2 | find_answer

    # c1 = context | find_answer
    # c = context | prompt | chat_llm  | StrOutputParser()

    return c2


def get_llm_chain(prompt, vector_store):
    chat_llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_CHATGPT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_type="azure",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0.5
    )
    retriever = vector_store.as_retriever()

    inputs = RunnableMap({
        "docs": RunnablePassthrough() | retriever,
        "question": RunnablePassthrough()
    })

    # Now we retrieve the documents
    context = RunnableMap({
        "context": RunnablePassthrough() | retriever | _combine_documents,
        "question": RunnablePassthrough(),
        "docs": RunnablePassthrough() | retriever,
    })

    context2 = RunnablePassthrough.assign(
        context=lambda x: _combine_documents(x["docs"]),
    )

    find_answer = RunnableMap({
        "answer":  prompt | chat_llm | StrOutputParser(),
        "docs": lambda x: x["docs"],
    })

    c2 = inputs | context2 | find_answer

    # c1 = context | find_answer
    # c = context | prompt | chat_llm  | StrOutputParser()

    return c2


def ask(question, llm_chain):
    answer = llm_chain.invoke(question)
    return answer


# generate a function to take a list of array and sort it by the first element
def sort_by_first_element(arr):
    return sorted(arr, key=lambda x: x[0])

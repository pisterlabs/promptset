# API import Section
from fastapi import FastAPI, UploadFile

# Langchain imports
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import SentenceTransformerEmbeddings

# Helper import
from typing import List
from PyPDF2 import PdfReader
import copy
import os


app = FastAPI(
    title="API Server",
    description="API Server for MoneyMaker",
    version="1.0",
)


def get_llm():
    """
    Initialize and return a ChatOpenAI language model instance.

    This function initializes a ChatOpenAI language model instance with specific configurations, including:
    - Temperature: Set to 0, which results in deterministic responses.
    - Maximum tokens: Set to 1000, limiting the response length.
    - Model name: Uses "gpt-3.5-turbo," indicating the model variant to be used.
    - Streaming mode: Enabled with 'streaming=True' for handling long conversations.

    Returns:
        llm (ChatOpenAI): An instance of the ChatOpenAI language model.
    """

    llm = ChatOpenAI(
        temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True
    )
    return llm


def get_pdf_texts(pdf_docs):
    """
    Extract text from a list of PDF documents.

    This function takes a list of PDF document file paths, reads each document, and extracts the text content
    from all pages of each PDF. It concatenates the text from all documents into a single string.

    Args:
        pdf_docs (list of str): A list of file paths to the PDF documents to extract text from.

    Returns:
        text (str): A string containing the concatenated text content from all the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Split a long text into smaller, overlapping chunks.

    This function takes a long text and splits it into smaller, overlapping chunks for more manageable processing.
    The text is divided into chunks with a specified chunk size and overlap. The resulting chunks are returned as a list.

    Args:
        text (str): The long text to be split into smaller chunks.

    Returns:
        chunks (list of str): A list of smaller text chunks.

    Note: The 'chunk_size' and 'chunk_overlap' parameters can be adjusted to control the chunk size and overlap
    according to your specific requirements.

    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def load_vectorstore(text_chunks):
    """
    Load or create a vector store for text chunks and add text data.

    This function checks if a directory named 'db' exists and contains a SQLite database file (with a .sqlite3
    extension). If such a file exists, it loads the existing vector store. If not, it creates a new vector store
    from the provided text chunks and saves it to the 'db' directory.

    Args:
        text_chunks (list of str): List of text chunks to be added to the vector store.

    Returns:
        success (bool): True if the vector store was successfully loaded or created, False otherwise.
    """

    if os.path.exists("./db") and os.path.isdir("./db"):
        sqlite_file = [file for file in os.listdir("./db") if file.endswith(".sqlite3")]
        if sqlite_file:
            embeddings = SentenceTransformerEmbeddings(
                model_name="multi-qa-MiniLM-L6-cos-v1"
            )
            vectorstore = Chroma(
                embedding_function=embeddings, persist_directory="./db"
            )
            vectorstore.add_texts(texts=text_chunks)

            return True
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="multi-qa-MiniLM-L6-cos-v1"
        )
        vectorstore = Chroma.from_texts(
            texts=text_chunks, embedding_function=embeddings, persist_directory="./db"
        )
        return False


def get_vectorstore():
    """
    Initialize and return a vector store for similarity-based text retrieval.

    This function initializes a vector store for text data, which can be used for similarity-based text retrieval.
    The vector store is created using SentenceTransformerEmbeddings and Chroma. It is configured for similarity search,
    with parameters such as 'k' for the number of similar items to retrieve.

    Returns:
        vectorstore (Chroma): A configured vector store for similarity-based text retrieval.
    """

    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    vectorstore = Chroma(
        persist_directory="./db",
        embedding_function=embeddings,
    ).as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return vectorstore


def get_prompt():
    """
    Generate a prompt template for answering questions with context and chat history.

    This function generates a prompt template that can be used for answering questions with context and chat history.
    The template includes placeholders for 'history,' 'context,' and 'question,' which should be provided when using
    the template to generate a prompt.

    Returns:
        prompt (PromptTemplate): A template for constructing prompts with placeholders for context, history, and question.

    """
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. If you dont know the answer, just say that you don't know, don't try to make up an answer:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    return prompt


def get_compressor():
    """
    Initialize and return a compressor with a contextual compression retriever.

    This function initializes a compressor using an LLMChainExtractor and a contextual compression retriever,
    which combines the compressor and a vector store for information retrieval.

    The components used in this function are as follows:
    - A vector store is loaded using the `get_vectorstore` function for similarity-based text retrieval.
    - An LLMChainExtractor is initialized using the `get_llm` function.
    - A contextual compression retriever is configured with the compressor and the vector store.

    Returns:
        compressor_retriever (ContextualCompressionRetriever): A configured compressor with a contextual compression retriever.

    """
    db = get_vectorstore()
    compressor = LLMChainExtractor.from_llm(llm=get_llm())
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=db
    )
    return compressor_retriever


def get_model():
    """
    Initialize and return a RetrievalQA model for answering questions with context and history.

    This function initializes a RetrievalQA model for answering questions with context and chat history. It combines
    several components to create the model:

    1. Compressor-Retriever: The compressor with a contextual compression retriever is obtained using the `get_compressor` function.

    2. Prompt: A prompt template is generated using the `get_prompt` function, which is used for constructing prompts with placeholders.

    3. RetrievalQA Model: The RetrievalQA model is initialized with the following components:
       - ChatOpenAI language model (LLM) obtained with the `get_llm` function.
       - Chain type set to "stuff."
       - Compressor-retriever as the retriever component.
       - Additional settings for verbosity and memory storage.

    Returns:
        model (RetrievalQA): A configured RetrievalQA model for answering questions with context and history.
    """
    compressor_retriever = get_compressor()
    prompt = get_prompt()
    model = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=compressor_retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferWindowMemory(
                memory_key="history",
                input_key="question",
                k=10,
                return_messages=True,
            ),
        },
    )
    return model


@app.get("/load_token")
async def load_token(token: str):
    """
    Load an OpenAI API token and initialize the question-answering model.

    This FastAPI endpoint is used to set the OpenAI API token and initialize the question-answering model
    when provided with a valid API token. The token is set as an environment variable, and the model is
    configured using the `get_model` function. If the token is successfully set, it returns a message indicating success.
    If the token is not provided or not valid, it returns a message indicating failure.

    Args:
        token (str): The OpenAI API token to be set as an environment variable.

    Returns:
        response (dict): A message indicating whether the token was successfully set or not. {"message": True} if successful,
        {"message": False} if not.
    """
    global qa
    os.environ["OPENAI_API_KEY"] = token
    print("TOKEN", token)
    if os.getenv("OPENAI_API_KEY"):
        qa = get_model()
        print("QA SET")
        return {"message": True}
    else:
        print("QA NOT SET")
        return {"message": False}


# LOAD SEVERAL PDFs ENDPOINT
@app.post("/load_pdfs")
async def load_pdfs(files: List[UploadFile]):
    """
    Load and process PDF documents to create a vector store.

    This FastAPI endpoint allows users to upload a list of PDF files. It processes the PDF documents to extract text
    content, divides the text into smaller chunks, and loads the chunks into a vector store for future retrieval.

    Args:
        files (List[UploadFile]): A list of uploaded PDF files.

    Returns:
        response (dict): A message indicating the success or failure of the PDF processing and vector store creation.
        - {"message": "PDF conversion successful, PLEASE RELOAD PAGE"} if successful.
        - {"message": "PDF conversion failed"} if unsuccessful.
    """

    archives = [f.file for f in files]

    text = get_pdf_texts(archives)
    # get chunks
    chunks = get_text_chunks(text)
    # load the vectorstore
    vector = load_vectorstore(chunks)
    if vector:
        return {"message": "PDF conversion succesful, PLEASE RELOAD PAGE"}
    else:
        return {"message": "PDF conversion failed"}


@app.get("/model")
async def model(question: str):
    """
    Use the question-answering model to answer a user's question.

    This FastAPI endpoint takes a user's question as input and uses the question-answering model to generate a response.

    Args:
        question (str): The user's question to be answered.

    Returns:
        result (dict): The response generated by the question-answering model.
    """
    result = qa({"query": question})
    result = copy.deepcopy(result["result"])
    return result

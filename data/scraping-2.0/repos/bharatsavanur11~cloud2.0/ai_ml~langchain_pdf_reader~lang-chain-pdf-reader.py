import os
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS

# Named constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def getOpenAiClient():
    """
    Retrieves the OpenAI client object.

    Returns:
        client : openai.api_client.OpenAIClient
            The OpenAI client object configured with the API key.

    """
    api_key = ''
    with open('api_key') as f:
        api_key = f.readline().strip('\n')
    client = OpenAI(api_key=api_key)
    os.environ["OPENAI_API_KEY"] = api_key
    return client


client = getOpenAiClient()


def readPdfDataAndReturnRawText():
    """
    Reads the content of a PDF file and returns the raw text.

    :return: The raw text extracted from the PDF file.
    """
    pdfReader = PdfReader('budget_speech.pdf')
    raw_text = ''
    for i, page in enumerate(pdfReader.pages):
        content = page.extract_text()
        if content:
            raw_text = raw_text + content
    return raw_text


raw_text = readPdfDataAndReturnRawText()


def split_text_character():
    """
    Split the raw_text into multiple chunks using a CharacterTextSplitter.

    :return: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                          length_function=len)
    texts = text_splitter.split_text(raw_text)
    return texts


texts = split_text_character()


def create_embeddings_and_search(texts):
    """
    Creates document embeddings for the given texts and performs a search using FAISS.

    :param texts: A list of texts for which embeddings are to be generated and searched.
    :type texts: list

    :return: The document search object generated using FAISS.
    :rtype: object
    """
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search


document_search = create_embeddings_and_search(texts)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

qna_chain = load_qa_chain(OpenAI(), chain_type='stuff')
query = "Vision for Amrit Kal"
docs = document_search.similarity_search(query)
output = qna_chain.run(input_documents=docs, question=query)
print(output)

import os
import boto3
import logging
import openai
import json
from PIL import Image
import pytesseract
from pypdf import PdfReader
from rq import Retry
from typing import Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings

from andes.models import Document, DocumentChatHistory
from andes.utils.config import UPLOAD_DIRECTORY
from andes.services.serialization import pickle_dump, pickle_load
from andes.services.rq import QUEUES
from andes.schemas.extraction_config import ExtractionConfigSchema
from andes.prompts import FIN_QA_PROMPT


QA_PROMPT = PromptTemplate(
    template=FIN_QA_PROMPT, 
    input_variables=["context", 'question'],
    template_format='jinja2'
)


S3_BUCKET = 'andes-chat-documents'
s3 = boto3.client('s3')


def create_document(filename: str):
    """
    create Document sqlalchemy model object
    """
    doc = Document(filename=filename)
    doc.save()
    return doc


def save_file(doc: Document, file):
    """
    saves the file to the uploads folder
    """
    logging.info(f"Saving Document {doc}")
    filepath = os.path.join(UPLOAD_DIRECTORY, doc.id, doc.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    return filepath


def get_document(id: str) -> Union[Document, None]:
    """
    get Document sqlalchemy model object
    """
    return Document.query.get(id)


def _run_ocr(fpath: str) -> str:
    """
    Perform OCR on a PDF or Image file
    """
    if fpath.lower().endswith('pdf'):
        reader = PdfReader(fpath)
        raw_document_text = '\n\n'.join([page.extract_text() for page in reader.pages])
    else:
        image = Image.open(fpath)
        raw_document_text = pytesseract.image_to_string(image)

    return raw_document_text


def _split_pdf(raw_document_text: str, chunk_size=4000, chunk_overlap=50) -> list[str]:
    """
    Pre-process PDF into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
        add_start_index = True,
    )

    splits = text_splitter.create_documents([raw_document_text])
    texts = [split.page_content for split in splits]
    return texts


def create_index(doc: Document):
    """
    create a langchain index for the document
    """
    logging.info(f"Started creating index for {doc.filename}")

    filepath = os.path.join(UPLOAD_DIRECTORY, doc.id, doc.filename)
    raw_document_text = _run_ocr(filepath)

    # save the raw document text to disk
    raw_document_text_path = os.path.join(UPLOAD_DIRECTORY, doc.id, 'raw_document_text.txt')
    with open(raw_document_text_path, 'w') as f:
        f.write(raw_document_text)

    # upload the file to s3
    file_name = f'{doc.id}.pdf'
    s3.upload_file(filepath, S3_BUCKET, file_name,
        ExtraArgs={
            'ContentType': 'application/pdf',
            'ContentDisposition': 'inline'
        }
    )

    # split the document into chunks
    doc_splits = _split_pdf(raw_document_text)

    # create a langchain index for each chunk
    logging.info(f"Building index for {doc.filename}")
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(doc_splits, embeddings)

    # save the index to disk
    index_path = os.path.join(UPLOAD_DIRECTORY, doc.id, 'index.pkl')
    pickle_dump(index, index_path)


def enqueue_index_gen(doc: Document):
    """
    enqueue the index generation task into a redis queue
    """
    QUEUES['index_gen'].enqueue(create_index, doc, retry=Retry(max=3))
    logging.info(f"Enqueued index generation for {doc.filename}")


def chat(doc: Document, message: str) -> str:
    # query openai on the langchain index

    # sanity checks
    if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, doc.id, 'index.pkl')):
        raise ValueError("Index does not exist for this document")
    
    assert message is not None, "Message cannot be empty"

    # load the index from disk
    index_path = os.path.join(UPLOAD_DIRECTORY, doc.id, 'index.pkl')
    index = pickle_load(index_path)

    # create a chain
    qa = RetrievalQA.from_chain_type(
        OpenAI(temperature=0), 
        chain_type="stuff", 
        retriever=index.as_retriever(),
        chain_type_kwargs={
            "prompt": QA_PROMPT
        }
    )

    # query GPT
    response = qa.run(message)

    # save the chat history
    DocumentChatHistory(
        document_id = doc.id,
        question = message,
        answer = response
    ).save()

    return response


def extract(doc: Document, config: ExtractionConfigSchema) -> str:
    # query openai on the langchain index

    # sanity checks
    if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, doc.id, 'index.pkl')):
        raise ValueError("Index does not exist for this document")
    
    assert config is not None, "Extraction config cannot be empty"

    # load OCR text from disk
    raw_document_text_path = os.path.join(UPLOAD_DIRECTORY, doc.id, 'raw_document_text.txt')
    with open(raw_document_text_path, 'r') as f:
        raw_document_text = f.read()

    # create functions to call using the config
    functions = [
        {
            'name': 'extract_all_key_information',
            'description': 'Extract all key information from the document',
            'parameters': {
                'type': 'object',
                # create a schema from the config
                'properties': {
                    x['label'] : {'type': 'string'} for x in config['entities']
                }
            }
        }
    ]

    # function calling using GPT 4
    response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [{'role': 'user', 'content': raw_document_text}],
        functions = functions,
        function_call = 'auto'
    )
    
    response = response['choices'][0]['message']

    # save raw response to disk
    raw_response_path = os.path.join(UPLOAD_DIRECTORY, doc.id, 'raw_openai_response.txt')
    with open(raw_response_path, 'w') as f:
        f.write(str(response))

    # parse function call input
    return json.loads(response['function_call']['arguments'])

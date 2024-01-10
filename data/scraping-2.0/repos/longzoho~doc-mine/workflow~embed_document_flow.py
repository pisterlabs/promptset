import json
import logging

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from prefect import task, flow

from data_types import FileStatus
from repository.documents import Documents
from repository.user_document import UserDocument
from util.file_util import get_file_path_by_key, file_exists, get_file_content
from util.path_util import bucket, embeddingdb_path, get_document_file_key
from worker.worker_decorator import conversation_asking_channel
from workflow.document_summary_flow import summary_document_flow

logger = logging.getLogger(__name__)

chunk_size = 1000
chunk_overlap = 200


@task(name='Get file key if exists else None')
def get_file_key_if_exists(hash_name: str, user_id: str) -> list[str]:
    file_key = get_document_file_key(hash_name=hash_name) if file_exists(bucket=bucket(),
                                                                         file_key=get_document_file_key(
                                                                             hash_name=hash_name)) else None
    if file_key is None:
        UserDocument(user_id=user_id, hash_name=hash_name).update_status(status=FileStatus.ERROR)
        Documents(hash_name=hash_name).update_status(status=FileStatus.ERROR)
    return file_key


@task(name='Check if chroma db exists')
def is_chroma_db_exists(hash_name: str, user_id) -> bool:
    result = file_exists(bucket=bucket(), file_key=f'{embeddingdb_path()}/{hash_name}')
    if result:
        UserDocument(user_id=user_id, hash_name=hash_name).update_status(status=FileStatus.EMBED_SAVED)
        Documents(hash_name=hash_name).update_status(status=FileStatus.EMBED_SAVED)
    return result


@task(name='Split documents to chunks')
def split_chunks(document_file_key) -> list[Document]:
    document = Document(**json.loads(get_file_content(bucket=bucket(), file_key=document_file_key)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents=[document])


@task(name='Embed chunks to chroma db')
def embed_chunk(hash_name: str, user_id: str, chunks: list[Document]):
    # embed document into chroma db
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": 'cuda'},
    )

    # embedd document into chroma db
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=get_file_path_by_key(bucket=bucket(), file_key=f'{embeddingdb_path()}/{hash_name}'))
    db.persist()
    UserDocument(user_id=user_id, hash_name=hash_name).update_status(status=FileStatus.EMBED_SAVED)
    Documents(hash_name=hash_name).update_status(status=FileStatus.EMBED_SAVED)


@task(name='Run summary document flow')
def run_summary_document_flow(hash_name: str, user_id: str):
    summary_document_flow(hash_name=hash_name, user_id=user_id)


@flow(name='Embed documents to chroma db')
@conversation_asking_channel
def embed_document_flow(hash_name: str, user_id: str, **kwargs):
    msg_channel = kwargs['msg_channel']
    document_file_key = get_file_key_if_exists(hash_name, user_id=user_id)
    if document_file_key is None:
        return
    if not is_chroma_db_exists(hash_name=hash_name, user_id=user_id):
        chunks = split_chunks(document_file_key=document_file_key)
        embed_chunk(hash_name=hash_name, user_id=user_id, chunks=chunks)
    msg_channel.basic_publish(exchange='ai_search_process_exchange', routing_key='summary_document_topic',
                              body=json.dumps({'hash_name': hash_name, 'user_id': user_id}))

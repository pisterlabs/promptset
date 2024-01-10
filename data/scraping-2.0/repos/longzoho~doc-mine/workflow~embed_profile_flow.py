import json
import logging

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from prefect import task, flow

from data_types import FileStatus, ProfileStatus
from repository.profiles import Profiles
from util.file_util import get_file_path_by_key, file_exists, get_file_content
from util.path_util import bucket, embeddingdb_path, get_document_file_key

logger = logging.getLogger(__name__)

chunk_size = 1000
chunk_overlap = 200


@task(name='Get list files from profile')
def list_profile_files(profile_id: str, user_id: str) -> list[str]:
    profile = Profiles(user_id=user_id, profile_id=profile_id).get_profile()
    return list(
        get_document_file_key(key) for key, document in profile.get('documents').items() if
        document.get('status') == FileStatus.EMBED_SAVED)


@task(name='Check if profile chroma db exists')
def is_chroma_db_exists(profile_id: str, user_id) -> bool:
    result = file_exists(bucket=bucket(), file_key=f'{embeddingdb_path()}/{profile_id}')
    if result:
        Profiles(user_id=user_id, profile_id=profile_id).update_status(status=ProfileStatus.EMBED_SAVED)
    return result


@task(name='Split profile documents to chunks')
def split_chunks(profile_id: str, user_id: str, file_paths: list[str]) -> list[Document]:
    documents = [Document(**json.loads(get_file_content(bucket=bucket(), file_key=file_path))) for file_path in
                 file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents=documents)


@task
def embed_chunk(profile_id: str, user_id: str, chunks: list[Document]):
    # embed document into chroma db
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": 'cuda'},
    )

    # embedd document into chroma db
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=get_file_path_by_key(bucket=bucket(), file_key=f'{embeddingdb_path()}/{profile_id}'))
    db.persist()
    Profiles(user_id=user_id, profile_id=profile_id).update_status(status=ProfileStatus.EMBED_SAVED)


@flow(name='Embed profile documents to chroma db')
def embed_profile_flow(profile_id: str, user_id: str):
    file_paths = list_profile_files(profile_id=profile_id, user_id=user_id)
    if file_paths is None:
        return
    if not is_chroma_db_exists(profile_id=profile_id, user_id=user_id):
        chunks = split_chunks(profile_id=profile_id, user_id=user_id, file_paths=file_paths)
        embed_chunk(profile_id=profile_id, user_id=user_id, chunks=chunks)


def main(profile_id, user_id):
    embed_profile_flow(profile_id=profile_id, user_id=user_id).run()

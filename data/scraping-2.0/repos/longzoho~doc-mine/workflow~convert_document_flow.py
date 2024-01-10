import json
import logging
from typing import Optional

from langchain.document_loaders import PDFMinerLoader
from langchain.schema import Document
from prefect import task, flow

from data_types import FileStatus, DocumentData
from repository.documents import Documents
from repository.user_document import UserDocument
from util.file_util import save_file, file_exists, get_file_path_by_key
from util.path_util import bucket, get_document_file_key, get_content_file_key
from worker.worker_decorator import conversation_asking_channel
from workflow.message_channel import RoutingKeys

logger = logging.getLogger(__name__)

MAP_LOADER = {'pdf': PDFMinerLoader}


@task(name='create user document')
def create_user_document(document_data: DocumentData, user_id: str):
    UserDocument(user_id=user_id, hash_name=document_data.hash_name).create_if_not_exists(
        name=document_data.name, status=FileStatus.CONTENT_SAVED)
    Documents(hash_name=document_data.hash_name).create_if_not_exists(status=FileStatus.CONTENT_SAVED)


@task(name='get raw content of file')
def load_document(document_data: DocumentData, user_id: str) -> Optional[Document]:
    [file_name, file_ext] = document_data.hash_name.split('__')
    loader: PDFMinerLoader | None = MAP_LOADER.get(file_ext)
    document_entity = Documents(hash_name=document_data.hash_name)
    user_document_entity = UserDocument(user_id=user_id, hash_name=document_data.hash_name)

    if loader is None:
        document_entity.update_status(status=FileStatus.ERROR)
        user_document_entity.update_status(status=FileStatus.ERROR)
        return None

    if file_exists(bucket=bucket(), file_key=get_document_file_key(hash_name=document_data.hash_name)):
        document_entity.update_status(status=FileStatus.DOCUMENT_SAVED)
        user_document_entity.update_status(status=FileStatus.DOCUMENT_SAVED)
        return None

    try:
        file_path = get_file_path_by_key(bucket=bucket(),
                                         file_key=get_content_file_key(hash_name=document_data.hash_name))
        result = loader(file_path=file_path).load()[0]
        return result
    except Exception as e:
        document_entity.update_status(status=FileStatus.ERROR)
        user_document_entity.update_status(status=FileStatus.ERROR)
        logger.error(f'Error loading file {file_name}: {e}')
        return None


@task(name='Save document to document folder')
def save_document(document: Document, document_data: DocumentData, user_id: str):
    if document is None:
        return
    save_file(bucket=bucket(), file_key=get_document_file_key(hash_name=document_data.hash_name),
              file_data=document.json())
    Documents(hash_name=document_data.hash_name).update_status(status=FileStatus.DOCUMENT_SAVED)
    UserDocument(user_id=user_id, hash_name=document_data.hash_name).update_status(status=FileStatus.DOCUMENT_SAVED)
    return document


@task(name='Publish to embed document flow')
@conversation_asking_channel
def publish_to_embed_document_flow(document_data: DocumentData, user_id: str, **kwargs):
    # embed_document_flow(document_data=document_data, user_id=user_id)
    msg_channel = kwargs['msg_channel']
    print(json.dumps({
        'hash_name': document_data.hash_name,
        'user_id': user_id}))
    msg_channel.basic_publish(exchange='document_embed_process_exchange', routing_key=RoutingKeys.embed_document_topic,
                              body=json.dumps({
                                  'hash_name': document_data.hash_name,
                                  'user_id': user_id}))


@flow
def convert_documents_flow(document_data: list[DocumentData], user_id: str):
    create_user_document_task = create_user_document.map(document_data=document_data, user_id=user_id)
    document = load_document.map(document_data=document_data, user_id=user_id, wait_for=[create_user_document_task])
    save_document_task = save_document.map(document=document, document_data=document_data, user_id=user_id)
    # wait for all document to be saved
    publish_to_embed_document_flow.map(document_data=document_data, user_id=user_id, wait_for=[save_document_task])

import os
from celery import Celery
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, VectorDBQA
from server.utils import upload_file_to_aws_bucket, decode_file

from server.models import db
from server.models.user import User
from server.models.user_session import UserSession
from server.new_core import PDFQA

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery.conf.task_default_queue = os.environ.get("CELERY_DEFAULT_QUEUE", "studyoracle")
celery.conf.accept_content = ['application/json', 'application/x-python-serialize', 'pickle']

@celery.task(name="doc")
def add_doc(filename, file):
    try:
        # decode the file from base64
        file = decode_file(filename, file)
        # upload the file to the AWS bucket
        file_url = upload_file_to_aws_bucket(filename, file)
        return file_url
    except Exception as e:
        print("Error uploading file: ", e)
        return False
        
    
@celery.task(name="ask", serializer='pickle')
def handle_message(session_data, query):
    # run the query
    try:
        answer = session_data.query(query)
        return answer
    except Exception as e:
        print("Error in message handler: ", e)
        return False
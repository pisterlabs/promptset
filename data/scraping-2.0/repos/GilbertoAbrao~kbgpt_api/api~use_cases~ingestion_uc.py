import os
import json


from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import boto3
from requests import post, Response

from core.configs import settings
from core.enumerations import VectorDatabaseEnum, FilePlaceEnum

from drivers.faiss_driver import faiss_pdf_ingestor
from drivers.pinecone_driver import pinecone_ingest





async def ingestion_uc(file_path: str, file_place: FilePlaceEnum, vector_store: VectorDatabaseEnum, index_name: str, callback_url: str = None) -> str:
    """[summary]

    Args:
        file_path ([type]): [description]
        index_name ([type]): [description]

    Returns:
        str: [description]
    """

    try:
        
        match file_place:

            case FilePlaceEnum.s3:

                # check if directory exists
                if not os.path.exists(settings.LOCAL_TEMP_FOLDER):
                    os.makedirs(settings.LOCAL_TEMP_FOLDER)

                # extract file name
                local_file_path = f"{settings.LOCAL_TEMP_FOLDER}/{file_path.split('/')[-1]}" 

                # remove / if file_path starts with /
                if file_path[0] == '/':
                    file_path = file_path[1:]

                # get file from s3
                s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
                s3.download_file(settings.AWS_S3_BUCKET_NAME, file_path, local_file_path)

            case FilePlaceEnum.local:
                
                local_file_path = file_path
                

        
        # extract type of file
        file_type = local_file_path.split('.')[-1]

        # check if file is pdf
        match file_type.lower():
            
            case 'pdf':
            
                loader = PyPDFLoader(file_path=local_file_path)
                document = loader.load()
            
            case 'txt':

                loader = TextLoader(file_path=local_file_path)
                document = loader.load()
            
            case 'docx':
                loader = Docx2txtLoader(file_path=local_file_path)
                document = loader.load()


            case _:
                raise Exception("File type not supported")

        # check if vector store is pinecone
        match vector_store:

            case VectorDatabaseEnum.pinecone:

                await pinecone_ingest(documents=document, index_name=index_name)
                
            case VectorDatabaseEnum.faiss:
                ...

        
        # call callback_url if exists
        if callback_url:
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            response: Response = post(callback_url, headers=headers, data=json.dumps({'status': 'done'}))

        # delete local file 
        os.remove(local_file_path)
    

    except Exception as e:
    
        if callback_url:
            response: Response = post(callback_url, data=json.dumps({'status': 'error'}))

        raise Exception(e)


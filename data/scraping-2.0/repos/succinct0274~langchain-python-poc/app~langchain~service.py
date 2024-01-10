from fastapi import APIRouter, Depends
from typing import Annotated, List, Union, Optional, Dict
from uuid import UUID
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header, Path
from app.service.langchain.model import get_langchain_model
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.document_loaders import UnstructuredAPIFileIOLoader
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationKGMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document
import uuid
from langchain.memory.chat_message_histories import PostgresChatMessageHistory
from app.database.crud.conversation_history import create_conversation_history, find_conversation_historys_by_conversation_id, exists_conversation_historys_by_conversation_id
from app.database.schema.conversation_history import ConversationHistoryCreate
from app.database.base import SessionLocal, get_session_local
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks, Response
import logging
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from app.service.langchain.callbacks.postgres_callback_handler import PostgresCallbackHandler
from sse_starlette import EventSourceResponse
from app.service.langchain.callbacks.queue_callback_handler import QueueCallbackHandler
from queue import Queue, Empty
from langchain.chat_models import ChatOpenAI
import pandas as pd
from langchain.agents import initialize_agent
from app.service.langchain.callbacks.agent_queue_callback_handler import AgentQueueCallbackHandler
from app.service.langchain.parsers.output.output_parser import CustomConvoOutputParser
from langchain.agents.load_tools import _LLM_TOOLS
from app.service.langchain.agents.panda_agent import create_pandas_dataframe_agent
from app.service.langchain.models.chat_open_ai_with_token_count import ChatOpenAIWithTokenCount
from bson import Binary
from langchain.schema.runnable import RunnableBranch
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from app.mongodb.crud.document import update_document_status_by_ids, create_document, acreate_document, find_document_by_conversation_id_and_filenames, find_document_by_conversation_id
import app.mongodb.crud.document as mongodb_service
from app.mongodb.schema.document import DocumentCreate
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import date
from langchain.memory import ConversationBufferMemory
import time
import base64
import mimetypes
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
import re
from langchain.document_loaders import PyMuPDFLoader
from io import BytesIO
import io
import tempfile
from llama_index import Document as LlamaIndexDocument
from app.langchain.llama_tools import LlamaIndexRetriever
from llama_index.indices.vector_store import VectorStoreIndex
from app.langchain.llama_tools import LlamaIndexPgVectorStore
from llama_index.storage import StorageContext
from llama_index.vector_stores.types import MetadataFilters, MetadataFilter, FilterCondition, FilterOperator


logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
DEFAULT_AI_GREETING_MESSAGE = 'Hi there, how can I help you?'
SUPPORTED_DOCUMENT_TYPES = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])
SHARED_CONVERSATION_ID = os.getenv('SHARED_KNOWLEDGE_BASE_UUID')

def _process_pdf_files_with_llamaindex(files: List[UploadFile], conversation_id: str):
    
    documents: List[LlamaIndexDocument] = []
    pdf_files = [file for file in files if file.content_type == 'application/pdf']

    for pdf in pdf_files:
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                binary_bytes = pdf.file.read()
                tmp.write(binary_bytes)
                tmp.flush()
                pdf.file.seek(0)

                from llama_index import download_loader
                PyMuPDFReader = download_loader("PyMuPDFReader")
                loader = PyMuPDFReader()
                docs = loader.load_data(path)

                for doc in docs:
                    doc.metadata['conversation_id'] = conversation_id

                documents.extend(docs)
        finally:
            os.remove(path)

    return documents

def _process_pdf_files(files: List[UploadFile], conversation_id: str) -> List[Document]:
    documents = []
    pdf_files = [file for file in files if file.content_type == 'application/pdf']
    
    for pdf in pdf_files:
        from langchain.text_splitter import CharacterTextSplitter

        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                binary_bytes = pdf.file.read()
                tmp.write(binary_bytes)
                tmp.flush()
                pdf.file.seek(0)
                loader = PyMuPDFLoader(path)
                # Attempt to parse file into docs without ocr
                docs = loader.load_and_split(CharacterTextSplitter(separator="\n",
                                                                   chunk_size=800,
                                                                   chunk_overlap=100,
                                                                   length_function=len))
                
                if len(docs) == 0:
                    loader = PyMuPDFLoader(path, extract_images=True)
                    docs = loader.load_and_split(CharacterTextSplitter(separator="\n",
                                                                   chunk_size=800,
                                                                   chunk_overlap=100,
                                                                   length_function=len))
                
        finally:
            os.remove(path)

        # loader = UnstructuredAPIFileIOLoader(pdf.file, url=os.getenv('UNSTRUCTURED_API_URL'), metadata_filename=pdf.filename)
        # docs = loader.load_and_split(text_splitter = CharacterTextSplitter(separator="\n",
        #                                                                    chunk_size=800,
        #                                                                    chunk_overlap=100,
        #                                                                    length_function=len)
        #                                                                    )
        for doc in docs:
            doc.metadata['conversation_id'] = conversation_id
            doc.metadata['filename'] = pdf.filename
            if 'doc_in' in doc.metadata:
                continue
            doc.metadata['doc_id'] = str(uuid.uuid4())
        documents.extend(docs)
    
    return documents

def initiate_conversation(db_session: Session):
    sid = uuid.uuid4()

    while exists_conversation_historys_by_conversation_id(db_session, sid):
        sid = uuid.uuid4()

    create_conversation_history(db_session, ConversationHistoryCreate(conversation_id=sid,
                                                                   human_message='Hi',
                                                                   ai_message=DEFAULT_AI_GREETING_MESSAGE,
                                                                   greeting=True))
    
    return {
        'conversation_id': sid,
        'ai_message': DEFAULT_AI_GREETING_MESSAGE,
    }    

def get_xlsx_dataframes(files: List[UploadFile]):
    xlsx_files = [file for file in files if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if not xlsx_files:
        return []

    dataframes = [pd.read_excel(excel.file) for excel in xlsx_files]
    return dataframes

def load_document_to_vector_store_with_llamaindex(files: List[UploadFile], conversation_id: str) -> LlamaIndexRetriever:
    # Load documents
    documents = _process_pdf_files_with_llamaindex(files, conversation_id) 

    # Create vector store, thereby storage context
    vector_store = LlamaIndexPgVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    retriever = LlamaIndexRetriever(index=index, 
                                    query_kwargs={
                                        'filters': MetadataFilters(filters=[MetadataFilter(key='conversation_id', value=conversation_id),
                                                                            MetadataFilter(key='conversation_id', value=SHARED_CONVERSATION_ID)])
                                    })
    
    return retriever

def load_document_to_vector_store(files: List[UploadFile], conversation_id: str):
    documents = _process_pdf_files(files, conversation_id)

    if len(documents) == 0:
        return
    
    # Make vector store asynchronous
    # hf_embedding = HuggingFaceEmbeddings(
    #     model_name='sentence-transformers/all-MiniLM-L6-v2',
    #     encode_kwargs={'normalize_embeddings': False}
    # )

    db = PGVector(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=OpenAIEmbeddings(), 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_name=conversation_id,
                              collection_metadata={'conversation_id': conversation_id})
    
    db.add_documents(documents)

def find_conversation_historys(db_session: Session, conversation_id: str):
    return find_conversation_historys_by_conversation_id(db_session, conversation_id)

def find_document(conversation_id: str, filenames: List[str]):
    return find_document_by_conversation_id_and_filenames(conversation_id, filenames)

def upload(files: List[UploadFile], conversation_id: str = None):
    filenames = [f.filename for f in files]
    existed = find_document(conversation_id, filenames)
    existed_filenames = set([persisted.filename for persisted in existed])
    res: List[Dict] = []
    for file in files:
        if file.filename in existed_filenames:
            continue

        status = 'uploaded'
        entity = DocumentCreate(content=Binary(file.file.read()), filename=file.filename, mime_type=file.content_type, conversation_id=conversation_id, status=status)
        file.file.seek(0)
        grid_out = create_document(entity)
        res.append({
            'file_id': str(grid_out._id),
            'filename': file.filename,
            'content_type': file.content_type,
            'file': file,
            'status': 'uploaded'
        })

    return res
    
def upload_and_load(files: List[UploadFile], conversation_id: str = None):
    
    uploaded_files = upload(files, conversation_id)
    load_document_to_vector_store([uploaded['file'] for uploaded in uploaded_files], conversation_id)

def conversate_with_llm(db_session: Session, 
                        question: str,
                        files: List[UploadFile],
                        metadata: Dict[str, any],
                        conversation_id: str,
                        llm: ChatOpenAI,
                        background_tasks: BackgroundTasks | None,
                        instruction: str | None = None):
    logger.info(f"Question: {question}")
    

    file_detail = []
    if metadata is None or 'attachment' not in metadata:
        # Process files uploaded into text
        for file in files:
            file_detail.append({'filename': file.filename, 'mime_type': file.content_type})
        
        upload_and_load(files, conversation_id)
    else:
        # Changed status of all uploaded files in metadata
        attachments = metadata['attachment']
        file_ids = []
        for attachment in attachments:
            file_ids.append(attachment['file_id'])
            file_detail.append({'filename': attachment['filename'], 'mime_type': attachment['content_type']})

        update_document_status_by_ids(file_ids, conversation_id)

        # Find all documents specified in metadata
        file_with_details = mongodb_service.find_binary_document_by_file_ids(file_ids)
        files = []
        for file in file_with_details:
            upload_file = UploadFile(io.BytesIO(file['data']), 
                                     size=len(file['data']), 
                                     filename=file['filename'], 
                                     headers={'content-type': file['content_type']}) 
            files.append(upload_file)
            
        load_document_to_vector_store_with_llamaindex(files, conversation_id)
        load_document_to_vector_store(files, conversation_id)

    db = PGVector(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                            embedding_function=OpenAIEmbeddings(), 
                            distance_strategy=DistanceStrategy.EUCLIDEAN,
                            collection_name=conversation_id,
                            collection_metadata={'conversation_id': conversation_id})

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key='chat_history', return_messages=True, verbose=True, output_key='answer')

    # Load conversation history from the database if corresponding header provided
    if conversation_id is not None:
        chat_records = find_conversation_historys(db_session, conversation_id)
        for record in chat_records:
            memory.save_context({'input': record.human_message}, {'answer': record.ai_message})

    queue = Queue()

    additional_args = {}
    if instruction is not None:
        # https://stackoverflow.com/questions/76175046/how-to-add-prompt-to-langchain-conversationalretrievalchain-chat-over-docs-with
        messages = [
            SystemMessagePromptTemplate.from_template(instruction),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        additional_args['prompt'] = ChatPromptTemplate.from_messages(messages)

    # Cosntruct retriever from llamaindex
    vector_store = LlamaIndexPgVectorStore()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = LlamaIndexRetriever(index=index, 
                                query_kwargs={
                                    'filters': MetadataFilters(filters=[
                                        MetadataFilter(key='conversation_id', value=conversation_id),
                                        MetadataFilter(key='conversation_id', value=SHARED_CONVERSATION_ID)
                                    ], condition=FilterCondition.OR)
                                })

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True, callbacks=[QueueCallbackHandler(queue), PostgresCallbackHandler(db_session, conversation_id)]), 
        retriever=retriever,
        # retriever=db.as_retriever(search_kwargs={
        #     'filter': { 'conversation_id': {"in": [os.getenv('SHARED_KNOWLEDGE_BASE_UUID'), conversation_id]} }
        # }),
        condense_question_llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True),
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs=additional_args
    )

    df = get_xlsx_dataframes(files)

    chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either `DataFrame` or `RetrievalQA`.

            - Choose `DataFrame` if the question involves any of the following:
            - Operations related to Excel files or spreadsheets, including but not limited to creating, editing, or analyzing Excel data.
            - Interactions with CSV files, which may involve reading, writing, modifying, or manipulating CSV data in any form.
            - Questions about DataFrames in programming languages, particularly Python (e.g., pandas DataFrame). This includes creating, manipulating, analyzing, or any operations specific to these data structures.

            - Choose `RetrievalQA` if the question pertains to:
            - General knowledge or information queries that do not explicitly involve programming or data manipulation.
            - Queries about documents or PDF files, including reading, processing, extracting, or interpreting information from these formats.
            - Specific questions related to non-programming aspects of document handling or information retrieval.

            - Guidelines for Classification:
            - Ensure the classification is based on the main focus of the question.
            - If a question overlaps between categories but has a clear primary focus, classify according to the primary focus.
            - Do not respond with more than one word to maintain clarity.

            User Question:
            <question>
            {question}
            </question>

            Classification:"""
        )
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    timestamp = str(int(time.time()))
    image_name = f'{conversation_id}.png'
    exported_chart_path = f'/export/chart/{date.today()}/{timestamp}'
    image_path = f'{exported_chart_path}/{image_name}'
    def run_with_panda_agent(question: str):
        question += f"""
        If you have plotted a chart, you must not show the chart but you must save it as {image_path}. 
        Create a directory with the path {exported_chart_path} before you save your image.
        Also, never include the path to image into your answer.
        """
        panda_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, verbose=True), 
                                                        df[0] if len(df) == 1 else df,
                                                        verbose=True, 
                                                        agent_executor_kwargs={'handle_parsing_errors': True},
                                                        # return_intermediate_steps=True,
                                                        return_direct=True,
                                                        agent_type=AgentType.OPENAI_FUNCTIONS,
                                                        memory=ConversationBufferMemory())
        
        # Manually create a directory first
        Path(exported_chart_path).mkdir(parents=True, exist_ok=True)

        answer = panda_agent({'input': question})['output']

        # Add one more chain to rephrase answer
        prompt = ChatPromptTemplate.from_template("Rephrase the following output so that python code and image name does not appear in your response: {output}")
        chain = prompt | ChatOpenAIWithTokenCount(temperature=0, verbose=True)
        output = chain.invoke({"output": answer})
        return output.content
    
    branch = RunnableBranch(
        (lambda x: "dataframe" in x["topic"].lower(), lambda x: run_with_panda_agent(x['question'])),
        lambda x: qa(x['question'])['answer']
    )

    full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch
    answer = full_chain.invoke({'question': question})

    import logging
    import sys

    # Uncomment to see debug logs
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    logger.info(f"Answer : {answer}")

    res = { 'text': answer }

    # Convert the image into base64 as part of the response
    output_media = []
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type, _ = mimetypes.guess_type(image_path)
            output_media.append({
                'content': encoded_string,
                'content_type': mime_type
            })
            res['image'] = output_media

    if background_tasks is not None:
        # Save current conversation message to the database
        background_tasks.add_task(create_conversation_history, db_session, ConversationHistoryCreate(conversation_id=conversation_id, 
                                                                                                     human_message=question, 
                                                                                                     ai_message=answer, 
                                                                                                     uploaded_file_detail=file_detail,
                                                                                                     responded_media=output_media))

    return res

def handle_websocket_request(body: Dict[str, any], 
                             llm: ChatOpenAIWithTokenCount,
                             background_tasks: BackgroundTasks,
                             x_conversation_id: Annotated[str, Header()],
                             db_session: Session=Depends(get_session_local)):
    question = body['question']
    files: List[UploadFile] = []
    if 'files' in body:
        file_payloads = body['files']
        # Convert file payload into UploadFile starlette objects
        for payload in file_payloads:
            base64_str: str = payload['base64']
            file_bytes: bytes = base64_str.encode('utf-8')
            file_bytes = base64.b64decode(file_bytes)
            content_type, _ = mimetypes.guess_type(payload['filename'])
            upload_file = UploadFile(io.BytesIO(file_bytes), size=len(file_bytes), filename=payload['filename'], headers={
                'content-type': content_type
            })
            files.append(upload_file)
            
    result = conversate_with_llm(db_session, question, files, x_conversation_id, llm, background_tasks)
    return result
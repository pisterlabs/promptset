"""API endpoint definitions"""
import os
import json
from typing import List
from fastapi import (
    APIRouter,
    Request,
    Body,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    UploadFile,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import SecretStr
import gotrue.errors


import schema
from log_configs import log
from core.auth.supabase import Supabase

# from core.auth.supabase import (admin_auth_check_decorator,
#     chatbot_auth_check_decorator, chatbot_get_labels_decorator)
from core.pipeline import ConversationPipeline, DataUploadPipeline
from core.vectordb.chroma import Chroma
from core.vectordb.postgres4langchain import Postgres
from core.embedding.openai import OpenAIEmbedding
from core.embedding.sentence_transformers import SentenceTransformerEmbedding
from custom_exceptions import PermissionException, GenericException, ChatErrorResponse

router = APIRouter()
templates = Jinja2Templates(directory="templates")

WS_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/chat")
DOMAIN = os.getenv("DOMAIN", "localhost:8000")
POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "admin")
POSTGRES_DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD", "secret")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST", "localhost")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT", "5432")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "adotbcollection")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "chromadb_store")
CHROMA_DB_COLLECTION = os.environ.get(
    "CHROMA_DB_COLLECTION", "adotbcollection")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_PATH = "./uploaded-files/"

# pylint: disable=fixme, too-many-branches, too-many-statements, too-many-locals

auth_service = Supabase()


@router.get(
    "/",
    response_class=HTMLResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["General", "UI"],
)
async def index(request: Request):
    """Landing page"""
    log.info("In index router")
    return templates.TemplateResponse(
        "index.html", {"request": request, "demo_url": f"http://{DOMAIN}/app"}
    )


@router.get(
    "/test",
    response_model=schema.APIInfoResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["General"],
)
async def get_root():
    """Landing page with basic info about the App"""
    log.info("In root endpoint")
    return {"message": "App is up and running"}


# @router.get("/ui",
#     response_class=HTMLResponse,
#     responses={
#         422: {"model": schema.APIErrorResponse},
#         403: {"model": schema.APIErrorResponse},
#         500: {"model": schema.APIErrorResponse}},
#     status_code=200, tags=["UI"])
# async def get_ui(request: Request):
#     '''The development UI using http for chat'''
#     log.info("In ui endpoint!!!")
#     return templates.TemplateResponse("chat-demo.html",
#         {"request": request, "ws_url": WS_URL, "demo_url":f"http://{DOMAIN}/ui",
#         "demo_url2":f"http://{DOMAIN}/ui2", "login_url":f"http://{DOMAIN}/login"})


@router.get(
    "/app",
    response_class=HTMLResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["UI"],
)
# @chatbot_auth_check_decorator
async def get_ui2(request: Request):
    """The development UI using http for chat"""
    log.info("In ui endpoint!!!")
    return templates.TemplateResponse(
        "chat-demo-postgres.html",
        {
            "request": request,
            "ws_url": WS_URL,
            "demo_url": f"http://{DOMAIN}/app",
            "login_url": f"http://{DOMAIN}/login",
        },
    )


@router.get(
    "/login",
    response_class=HTMLResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["UI"],
)
async def get_login(request: Request):
    """The development login UI"""
    log.info("In login endpoint!!!")
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "ws_url": WS_URL,
            "demo_url": f"http://{DOMAIN}/ui",
            "demo_url2": f"http://{DOMAIN}/ui2",
        },
    )


def compose_vector_db_args(db_type, settings, embedding_config):
    """Convert the API params or default values, to args to initializing the DB"""
    vectordb_args = {}
    if settings.dbHostnPort:
        vectordb_args["host_n_port"] = settings.dbHostnPort
        parts = settings.dbHostnPort.split(":")
        vectordb_args["host"] = "".join(parts[:-1])
        vectordb_args["port"] = parts[-1]
    elif db_type == schema.DatabaseType.POSTGRES:
        vectordb_args["host_n_port"] = f"{POSTGRES_DB_HOST}:{POSTGRES_DB_PORT}"
        vectordb_args["host"] = POSTGRES_DB_HOST
        vectordb_args["port"] = POSTGRES_DB_PORT
    if settings.dbUser:
        vectordb_args["user"] = settings.dbUser
    elif db_type == schema.DatabaseType.POSTGRES:
        vectordb_args["user"] = POSTGRES_DB_USER
    if settings.dbPassword:
        vectordb_args["password"] = settings.dbPassword.get_secret_value()
    elif db_type == schema.DatabaseType.POSTGRES:
        vectordb_args["password"] = POSTGRES_DB_PASSWORD
    if settings.dbPath:
        vectordb_args["path"] = settings.dbPath
    elif db_type == schema.DatabaseType.CHROMA:
        vectordb_args["path"] = CHROMA_DB_PATH
    if settings.collectionName:
        vectordb_args["collection_name"] = settings.collectionName
    elif db_type == schema.DatabaseType.CHROMA:
        vectordb_args["collection_name"] = CHROMA_DB_COLLECTION
    elif db_type == schema.DatabaseType.POSTGRES:
        vectordb_args["collection_name"] = POSTGRES_DB_NAME
    if db_type == schema.DatabaseType.POSTGRES:
        log.info(
            "Because the db is Postgres, and embedding dimension size must be hard-coded, "
            + "setting embedding type to %s",
            embedding_config.embeddingType,
        )
        if embedding_config.embeddingType == schema.EmbeddingType.HUGGINGFACE_DEFAULT:
            vectordb_args["embedding"] = SentenceTransformerEmbedding()
        elif (
            embedding_config.embeddingType
            == schema.EmbeddingType.HUGGINGFACE_MULTILINGUAL
        ):
            vectordb_args["embedding"] = SentenceTransformerEmbedding(
                model="sentence-transformers/LaBSE"
            )
        elif embedding_config.embeddingType == schema.EmbeddingType.OPENAI:
            vectordb_args["embedding"] = OpenAIEmbedding()
        else:
            raise GenericException(
                "This embedding type is not supported (yet)!")

    return vectordb_args


@router.websocket("/chat")
@auth_service.chatbot_auth_check_decorator
@auth_service.chatbot_get_labels_decorator
async def websocket_chat_endpoint(
    websocket: WebSocket,
    settings=Depends(schema.ChatPipelineSelector),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
    labels: List[str] = Query(
        ["ESV-Bible"], desc="The document sets to be used for answering questions"
    ),
):
    """The http chat endpoint"""
    log.info("In chat endpoint!!!")
    if token:
        log.info("User, connecting with token, %s", token)
    await websocket.accept()
    chat_stack = ConversationPipeline(user="XXX", labels=labels)

    vectordb_args = compose_vector_db_args(
        settings.vectordbType,
        settings,
        schema.EmbeddingSelector(embeddingType=settings.embeddingType),
    )
    vectordb_args["labels"] = labels

    chat_stack.set_vectordb(settings.vectordbType, **vectordb_args)
    llm_args = {}
    if settings.llmApiKey:
        llm_args["api_key"] = settings.llmApiKey
    if settings.llmModelName:
        llm_args["model"] = settings.llmModelName
    chat_stack.set_llm_framework(
        settings.llmFrameworkType, vectordb=chat_stack.vectordb, **llm_args
    )
    chat_stack.set_transcription_framework(settings.transcriptionFrameworkType)

    # Not implemented using custom embeddings

    while True:
        try:
            # Receive and send back the client message
            received_bytes = await websocket.receive_bytes()
            try:  # Try treating the bytes as text
                received_json = received_bytes.decode("utf-8")
                received_dict = json.loads(received_json)
                log.info("Text received")
                if received_dict.get("type") == "reset":
                    chat_stack.chat_history = []
                    await websocket.send_json(
                        {
                            "sender": "Bot",
                            "message": "Let's start a new conversation! What's your question?",
                            "sources": [],
                        }
                    )
                    question = ""
                else:
                    question = received_dict.get("message")
            except UnicodeDecodeError:  # If that fails, treat it as audio
                log.info("Audio file received")
                question = chat_stack.transcription_framework.transcribe_audio(
                    received_bytes
                )
                start_human_q = schema.BotResponse(
                    sender=schema.SenderType.USER,
                    message=question,
                    type=schema.ChatResponseType.QUESTION,
                    sources=[],
                    media=[],
                )
                await websocket.send_json(start_human_q.dict())

            if len(question) > 0:
                bot_response = chat_stack.llm_framework.generate_text(
                    query=question, chat_history=chat_stack.chat_history
                )
                log.debug(
                    "Human: {0}\nBot:{1}\nSources:{2}\n\n".format(
                        question,
                        bot_response['answer'],
                        [item.metadata['source']
                            for item in bot_response['source_documents']]
                    )
                )

                chat_stack.chat_history.append(
                    (
                        bot_response["question"],
                        bot_response["answer"]
                    )
                )

                # Construct a response
                start_resp = schema.BotResponse(
                    sender=schema.SenderType.BOT,
                    message=bot_response["answer"],
                    type=schema.ChatResponseType.ANSWER,
                    sources=[
                        item.metadata["source"]
                        for item in bot_response["source_documents"]
                    ],
                    media=[],
                )
                await websocket.send_json(start_resp.dict())
        except ChatErrorResponse as exe:
            resp = schema.BotResponse(
                sender=schema.SenderType.BOT,
                message="Sorry, I am unable to answer that. " + exe.detail,
                type=schema.ChatResponseType.ERROR,
            )
            await websocket.send_json(resp.dict())
            break
        except WebSocketDisconnect:
            if isinstance(chat_stack.vectordb, Chroma):
                chat_stack.vectordb.db_client.persist()
            log.info("websocket disconnect")
            break
        except Exception as exe:  # pylint: disable=broad-exception-caught
            log.exception(exe)
            resp = schema.BotResponse(
                sender=schema.SenderType.BOT,
                message="Sorry, something went wrong. Try again.",
                type=schema.ChatResponseType.ERROR,
            )
            await websocket.send_json(resp.dict())


@router.post(
    "/upload/sentences",
    response_model=schema.APIInfoResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=201,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def upload_sentences(
    document_objs: List[schema.Document] = Body(
        ..., desc="List of pre-processed sentences"
    ),
    vectordb_type: schema.DatabaseType = Query(schema.DatabaseType.CHROMA),
    vectordb_config: schema.DBSelector = Depends(schema.DBSelector),
    embedding_config: schema.EmbeddingSelector = Depends(
        schema.EmbeddingSelector),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """* Upload of any kind of data that has been pre-processed as list of sentences.
    * Vectorises the text using OpenAI embdedding (or the one set in chroma DB settings).
    * Keeps other details, sourceTag, link, and media as metadata in vector store
    * embedding_type: optional for ChromaDB. For Postgres, if none, will use OpenAIEmbedding
    """
    log.info("Access token used:%s", token)
    data_stack = DataUploadPipeline()
    vectordb_args = compose_vector_db_args(
        vectordb_type, vectordb_config, embedding_config
    )
    data_stack.set_vectordb(vectordb_type, **vectordb_args)
    data_stack.set_embedding(
        embedding_config.embeddingType,
        embedding_config.embeddingApiKey,
        embedding_config.embeddingModelName,
    )
    # FIXME: This may have to be a background job!!!
    data_stack.embedding.get_embeddings(doc_list=document_objs)

    # FIXME: This may have to be a background job!!!
    data_stack.vectordb.add_to_collection(docs=document_objs)
    return {"message": "Documents added to DB"}


@router.post(
    "/upload/text-file",
    response_model=schema.APIInfoResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=201,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def upload_text_file(  # pylint: disable=too-many-arguments
    file_obj: UploadFile,
    label: str = Query(
        ..., desc="The label for the set of documents for access based filtering"
    ),
    file_processor_type: schema.FileProcessorType = Query(
        schema.FileProcessorType.LANGCHAIN
    ),
    vectordb_type: schema.DatabaseType = Query(schema.DatabaseType.CHROMA),
    vectordb_config: schema.DBSelector = Depends(schema.DBSelector),
    embedding_config: schema.EmbeddingSelector = Depends(
        schema.EmbeddingSelector),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """* Upload of any kind text files like .md, .txt etc.
    * Splits the whole document into smaller chunks using the selected file_processor
    * Vectorises the text using OpenAI embdedding (or the one set in chroma DB settings).
    * Keeps other details, sourceTag, link, and media as metadata in vector store
    * embedding_type: optional for ChromaDB. For Postgres, if none, will use OpenAIEmbedding
    """
    log.info("Access token used: %s", token)
    vectordb_args = compose_vector_db_args(
        vectordb_type, vectordb_config, embedding_config
    )
    data_stack = DataUploadPipeline()
    data_stack.set_vectordb(vectordb_type, **vectordb_args)

    data_stack.set_file_processor(file_processor_type)

    if not os.path.exists(UPLOAD_PATH):
        os.mkdir(UPLOAD_PATH)
    with open(f"{UPLOAD_PATH}{file_obj.filename}", "w", encoding="utf-8") as tfp:
        tfp.write(file_obj.file.read().decode("utf-8"))

    # FIXME: This may have to be a background job!!!
    docs = data_stack.file_processor.process_file(
        file_path=f"{UPLOAD_PATH}{file_obj.filename}",
        file_type=schema.FileType.TEXT,
        label=label,
        name="".join(file_obj.filename.split(".")[:-1]),
    )
    data_stack.set_embedding(
        embedding_config.embeddingType,
        embedding_config.embeddingApiKey,
        embedding_config.embeddingModelName,
    )
    # FIXME: This may have to be a background job!!!
    data_stack.embedding.get_embeddings(doc_list=docs)
    data_stack.vectordb.add_to_collection(docs=docs)
    return {"message": "Documents added to DB"}


@router.post(
    "/upload/pdf-file",
    response_model=schema.APIInfoResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=201,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def upload_pdf_file(  # pylint: disable=too-many-arguments
    file_obj: UploadFile,
    label: str = Query(
        ..., desc="The label for the set of documents for access based filtering"
    ),
    file_processor_type: schema.FileProcessorType = Query(
        schema.FileProcessorType.LANGCHAIN
    ),
    vectordb_type: schema.DatabaseType = Query(
        schema.DatabaseType.CHROMA
    ),
    vectordb_config: schema.DBSelector = Depends(
        schema.DBSelector
    ),
    embedding_config: schema.EmbeddingSelector = Depends(
        schema.EmbeddingSelector
    ),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """* Upload of a pdf file.
    * Splits the whole document into smaller chunks using the selected file_processor
    * Vectorises the text using OpenAI embdedding (or the one set in chroma DB settings).
    * Keeps other details, sourceTag, link, and media as metadata in vector store
    * embedding_type: optional for ChromaDB. For Postgres, if none, will use OpenAIEmbedding
    """
    log.info("Access token used: %s", token)

    vectordb_args = compose_vector_db_args(vectordb_type, vectordb_config, embedding_config)
    data_stack = DataUploadPipeline()
    data_stack.set_vectordb(vectordb_type, **vectordb_args)

    data_stack.set_file_processor(file_processor_type)

    if not os.path.exists(UPLOAD_PATH):
        os.mkdir(UPLOAD_PATH)

    data = await file_obj.read()
    with open(f"{UPLOAD_PATH}{file_obj.filename}", "wb") as tfp:
        tfp.write(data)

    # FIXME: This may have to be a background job!!!
    docs = data_stack.file_processor.process_file(
        file_path=f"{UPLOAD_PATH}{file_obj.filename}",
        file_type=schema.FileType.PDF,
        label=label,
        name="".join(file_obj.filename.split(".")[:-1]),
    )
    data_stack.set_embedding(
        embedding_config.embeddingType,
        embedding_config.embeddingApiKey,
        embedding_config.embeddingModelName,
    )
    # FIXME: This may have to be a background job!!!
    data_stack.embedding.get_embeddings(doc_list=docs)
    data_stack.vectordb.add_to_collection(docs=docs)
    return {"message": "Documents added to DB"}

@router.post(
    "/upload/csv-file",
    response_model=schema.APIInfoResponse,
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=201,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def upload_csv_file(  # pylint: disable=too-many-arguments
    file_obj: UploadFile,
    col_delimiter: schema.CsvColDelimiter = Query(
        schema.CsvColDelimiter.COMMA, desc="Seperator used in input file"
    ),
    vectordb_type: schema.DatabaseType = Query(schema.DatabaseType.CHROMA),
    vectordb_config: schema.DBSelector = Depends(schema.DBSelector),
    embedding_config: schema.EmbeddingSelector = Depends(
        schema.EmbeddingSelector),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """* Upload CSV with fields (id, text, label, links, medialinks).
    * Vectorises the text using OpenAI embdedding (or the one set in chroma DB settings).
    * Keeps other details, sourceTag, link, and media as metadata in vector store
    * embedding_type: optional for ChromaDB. For Postgres, if none, will use OpenAIEmbedding
    """
    log.info("Access token used: %s", token)
    vectordb_args = compose_vector_db_args(
        vectordb_type, vectordb_config, embedding_config
    )
    data_stack = DataUploadPipeline()
    data_stack.set_vectordb(vectordb_type, **vectordb_args)
    if not os.path.exists(UPLOAD_PATH):
        os.mkdir(UPLOAD_PATH)
    with open(f"{UPLOAD_PATH}{file_obj.filename}", "w", encoding="utf-8") as tfp:
        tfp.write(file_obj.file.read().decode("utf-8"))

    # FIXME: This may have to be a background job!!!
    if col_delimiter == schema.CsvColDelimiter.COMMA:
        col_delimiter = ","
    elif col_delimiter == schema.CsvColDelimiter.TAB:
        col_delimiter = "\t"
    docs = data_stack.file_processor.process_file(
        file_path = f"{UPLOAD_PATH}{file_obj.filename}",
        file_type = schema.FileType.CSV,
        col_delimiter = col_delimiter,
        )
    data_stack.set_embedding(
        embedding_config.embeddingType,
        embedding_config.embeddingApiKey,
        embedding_config.embeddingModelName,
    )
    # FIXME: This may have to be a background job!!!
    data_stack.embedding.get_embeddings(doc_list=docs)
    data_stack.vectordb.add_to_collection(docs=docs)
    return {"message": "Documents added to DB"}


@router.get(
    "/job/{job_id}",
    response_model=schema.Job,
    responses={
        404: {"model": schema.APIErrorResponse},
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def check_job_status(
    job_id: int = Path(...),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """Returns the status of background jobs like upload-documemts"""
    log.info("Access token used:%s", token)
    print(job_id)
    return {"jobId": "10001", "status": schema.JobStatus.QUEUED}


@router.get(
    "/source-labels",
    response_model=List[str],
    responses={
        422: {"model": schema.APIErrorResponse},
        403: {"model": schema.APIErrorResponse},
        500: {"model": schema.APIErrorResponse},
    },
    status_code=200,
    tags=["Data Management"],
)
@auth_service.admin_auth_check_decorator
async def get_source_tags(
    db_type: schema.DatabaseType = schema.DatabaseType.CHROMA,
    settings: schema.DBSelector = Depends(schema.DBSelector),
    embedding_config: schema.EmbeddingSelector = Depends(
        schema.EmbeddingSelector),
    token: SecretStr = Query(
        None, desc="Optional access token to be used if user accounts not present"
    ),
):
    """Returns the distinct set of source tags available in chorma DB"""
    log.debug(
        "host:port:%s, path:%s, collection:%s",
        settings.dbHostnPort,
        settings.dbPath,
        settings.collectionName,
    )
    log.info("Access token used: %s", token)
    args = compose_vector_db_args(db_type, settings, embedding_config)

    if db_type == schema.DatabaseType.CHROMA:
        vectordb = Chroma(**args)
    elif db_type == schema.DatabaseType.POSTGRES:
        vectordb = Postgres(**args)
    else:
        raise GenericException("This database type is not supported (yet)!")

    return vectordb.get_available_labels()


@router.post("/login")
async def login(
    email=Form(..., desc="Email of the user"),
    password=Form(..., desc="Password of the user"),
):
    """Signs in a user"""
    try:
        data = auth_service.conn.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except gotrue.errors.AuthApiError as exe:
        log.error(f"We have an error: {exe}")
        if str(exe) == "Email not confirmed":
            print("It's an email not confirmed error")
            raise HTTPException(
                status_code=401,
                detail="The user email hasn't been confirmed. "
                + "Please confirm your email and then try to log in again.",
            ) from exe

        raise PermissionException(
            "Unauthorized access. Invalid credentials.") from exe

    return {
        "message": "User logged in successfully",
        "access_token": data.session.access_token,
    }


@router.post("/logout")
async def logout():
    """Signs out a user"""
    auth_service.conn.auth.sign_out()

    return {
        "message": "User logged out successfully",
        "next_url": f"http://{DOMAIN}/login",
    }


@router.post("/signup")
async def signup(
    email=Form(..., desc="Email of the user"),
    password=Form(..., desc="Password of the user"),
):
    """Signs up a new user"""
    try:
        auth_service.conn.auth.sign_up(
            {
                "email": email,
                "password": password,
            }
        )
    except gotrue.errors.AuthApiError as exce:
        raise PermissionException("Sign up error") from exce

    return {
        "message": "Please check your email to confirm your account.",
        "access_token": None,
    }

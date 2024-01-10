from langchain.chat_models import ChatOpenAI
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    GPTVectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
import os, logging

# Set base directory and load environment variables
from pathlib import Path
from dotenv import load_dotenv

# Set base directory and load environment variables -- IT is must
BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def process_initializers():  # only initialize variables
    # Configuration parameters
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    logging.info("Inside process_initializers")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    try:
        # Initialize prompt_helper and ChatOpenAI
        prompt_helper = PromptHelper(
            max_input_size,
            num_outputs,
            max_chunk_overlap,
            chunk_size_limit=chunk_size_limit,
        )
        # llm = ChatOpenAI(
        #     openai_api_key=openai_api_key, temperature=0.3, model_name="gpt-4"
        # )
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-0613",
            request_timeout=12,
            max_retries=2,
            # max_tokens=150,
            # model_kwargs={"stop": "\n"},
            # model_kwargs={"messages": last_chats},
        )

        llm_predictor = LLMPredictor(llm=llm)
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )
        return service_context
    except Exception as e:
        logging.error(e)
        raise


# Convert uploaded file into index and store in target
def construct_index() -> None:
    logging.info("inside construct_index")
    cwd = os.getcwd()
    file_repo_path = os.path.join(cwd, "file_repo")
    index_repo_path = os.path.join(cwd, "index_repo")

    try:
        service_context = process_initializers()  # calling method
        documents = SimpleDirectoryReader(file_repo_path).load_data()
        # when first building the index
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        # handle index for storage and retrieval
        index.storage_context.persist(persist_dir=index_repo_path)  # -> store in db
    except Exception as e:
        logging.error(e)
        raise


# load index file for interaction
def load_query_engine():
    logging.info("inside query_engine")
    # define the relative path to the index_repo folder
    cwd = os.getcwd()
    index_repo_path = os.path.join(cwd, "index_repo")

    try:
        service_context = process_initializers()
        storage_context = StorageContext.from_defaults(
            persist_dir=index_repo_path
        )  # <- load from db

        index = load_index_from_storage(
            service_context=service_context, storage_context=storage_context
        )
        # prompt handling

        query_engine = index.as_query_engine()
        # query_engine = index.as_retriever()

        return query_engine
    except Exception as e:
        logging.error(e)
        raise


# # additions 7 september
# def load_index_vectors():
#     logging.info("inside vector")
#     # define the relative path to the index_repo folder
#     cwd = os.getcwd()
#     index_repo_path = os.path.join(cwd, "index_repo")

#     try:
#         service_context = process_initializers()
#         storage_context = StorageContext.from_defaults(
#             persist_dir=index_repo_path
#         )  # <- load from db

#         index = load_index_from_storage(
#             service_context=service_context, storage_context=storage_context
#         )
#         return index
#     except Exception as e:
#         logging.error(e)
#         raise

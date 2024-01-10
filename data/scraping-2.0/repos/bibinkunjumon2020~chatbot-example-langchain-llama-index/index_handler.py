# index_handler.py
import os, logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
)
from llama_index.llms import OpenAI
import openai
from langchain.chat_models import ChatOpenAI

# Set base directory and load environment variables -- IT is must
BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]

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
            chunk_overlap_ratio=0.2,
            chunk_size_limit=chunk_size_limit,
        )
        # llm = ChatOpenAI(
        #     openai_api_key=openai_api_key, temperature=0.3, model_name="gpt-4"
        # )
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-0613",
            request_timeout=20,
            max_retries=2,
            max_tokens=200,  # max token to generate
            # model_kwargs={"stop": "\n"},
            # model_kwargs={"messages": last_chats},
        )
        # llm = OpenAI(api_key=openai_api_key,temperature=0,model="gpt-3.5-turbo-0613",\
        #              additional_kwargs={"request_timeout":12},max_retries=2)

        llm_predictor = LLMPredictor(llm=llm)
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )
        return service_context
    except Exception as e:
        logging.error(str(e))
        raise


def store_index():
    try:
        # get the current working directory
        cwd = os.getcwd()
        # define the relative path to the index_repo folder
        file_repo_path = os.path.join(cwd, "file_repo")
        index_repo_path = os.path.join(cwd, "index_repo")
        service_context = process_initializers()  # calling method

        documents = SimpleDirectoryReader(file_repo_path).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents=documents, service_context=service_context
        )
        # index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_repo_path)
    except Exception as e:
        logging.error(str(e))   
        raise


def load_index():
    try:
        # get the current working directory
        cwd = os.getcwd()
        # define the relative path to the index_repo folder
        index_repo_path = os.path.join(cwd, "index_repo")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_repo_path)
        service_context = process_initializers()  # calling method

        # load index
        index = load_index_from_storage(
            storage_context=storage_context, service_context=service_context
        )
        return index
    except Exception as e:
        logging.error(str(e))
        raise

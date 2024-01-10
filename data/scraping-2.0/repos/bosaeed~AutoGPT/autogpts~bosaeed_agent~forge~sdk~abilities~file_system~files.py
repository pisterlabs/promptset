from typing import List

from ..registry import ability

from ...forge_log import ForgeLogger

from forge.sdk import (

    chat_completion_request,
)

import os

import pprint

LOGGER = ForgeLogger(__name__)

import csv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import weaviate

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]



# @ability(
#     name="list_files",
#     description="List files in a directory use to know name of avaliable files",
#     parameters=[
#         {
#             "name": "path",
#             "description": "Path to the directory ,use '/' for current directory",
#             "type": "string",
#             "required": True,
#         }
#     ],
#     output_type="list[str]",
# )
# async def list_files(agent, task_id: str, path: str) -> List[str]:
#     """
#     List files in a workspace directory
#     """
#     try:
#         output = agent.workspace.list(task_id=task_id, path=path)
#     except Exception as e:
#         return "no any file exist"

#     if output:
#         return "avaliable files: " + (" , ".join(str(element) for element in output))
#     return "no any file exist"


@ability(
    name="write_file",
    description="create file and Write data to it",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_name: str, data: bytes ) -> str:
    """
    Write data to a file
    """
    

    # if(agent.workspace.exists( task_id, file_name)):
    #     return f"file {file_name} already exist"

    if(".py" in file_name and await is_correct_python(data) != True):
        return "provided data in not valid python code"

    if isinstance(data, str):
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_name, data=data)
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_name.split("/")[-1],
        relative_path=file_name,
        agent_created=True,
    )

    return f"writing to file done successfully"


@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "query",
            "description": "query for information needed",
            "type": "string",
            "required": False,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str , query: str) -> bytes:
    """
    Read data from a file
    """
    try:
        output = agent.workspace.read(task_id=task_id, path=file_path).decode()
        # output = file.decode()

        if len(output) > 2000: 

            if(".csv" in file_path):
                documents = read_csv_file(agent , task_id , file_path) 

                te = ""
                output = ""
                for line in documents:
                    # print(line)
                    te += line + "\n"
                    if(len(te) >= 10000):
                        output += await summeraize_texts(agent,te,query)
                        te = ""

                if(len(te) > 0):
                    output += await summeraize_texts(agent,te,query)
                    te = ""


            else:

                weaviate_client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY))
                embeddings = OpenAIEmbeddings()

                LOGGER.info("start weaviate client")
                # LOGGER.info(output)
                # loader = TextLoader("../../modules/state_of_the_union.txt")
                # documents = loader.load()

                
                text_splitter = RecursiveCharacterTextSplitter(
                    # Set a really small chunk size, just to show.
                    chunk_size = 1000,
                    chunk_overlap  = 0,
                    length_function = len,
                    add_start_index = True,
                )
                LOGGER.info("**********************split*********************")

               
                documents = text_splitter.split_text(output)

                LOGGER.info("******************document generated************")
                # LOGGER.info(documents)
                vectorstore = Weaviate.from_texts(documents, embeddings, client=weaviate_client, by_text=False)
                LOGGER.info("****************vectorestore added***************")

                task_query = agent.current_task

                query_embedding = embeddings.embed_query(query)

                task_query_embedding = embeddings.embed_query(task_query)
                LOGGER.info("*****************query embed done*************")
                docs = vectorstore.similarity_search_by_vector(query_embedding  , k=5)

                task_docs = vectorstore.similarity_search_by_vector(task_query_embedding , k=5)

                LOGGER.info("*****************similarity done***********")
                # docs = vectorstore.similarity_search(query)

                output = ""
                for doc in docs:
                    output += doc.page_content + " \n"

                task_output = ""
                for doc in task_docs:
                    task_output += doc.page_content + " \n"
                LOGGER.info(output)
                LOGGER.info(task_output)



    except Exception as e:
        output = f"File Not found may need create one first {e}"
        
    return output


def read_csv_file( agent , task_id , path) :
    docs = []
    with open(agent.workspace._resolve_path(task_id, path), newline="") as csvfile:
        # return f.read()
        
        metadata = []
        csv_reader = csv.DictReader(csvfile , delimiter="\t")  # type: ignore
        for i, row in enumerate(csv_reader):

            content = ",".join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                # if k not in self.metadata_columns
            )

            docs.append(content)

    return docs


async def summeraize_texts(agent, text ,query):

    model =  os.getenv('FAST_LLM', "gpt-3.5-turbo")
    # agent.prompt_engine = PromptEngine("gpt-3.5-turbo" , agent.debug)
    system_prompt = agent.prompt_engine.load_prompt("summerize-system")


    task_kwargs = {
        "query": query,
        "text": text,
        
    }
    # LOG.info(pprint.pformat(task_kwargs))
    # Then, load the task prompt with the designated parameters
    task_prompt = agent.prompt_engine.load_prompt("summerize-user", **task_kwargs)
    #messages list:
    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ]
    
    try:

        chat_completion_kwargs = {
            "messages": messages,
            "model": model,

        }
        # Make the chat completion request and parse the response
        LOGGER.info(pprint.pformat(chat_completion_kwargs))
        chat_response = await chat_completion_request(**chat_completion_kwargs)

        LOGGER.info(pprint.pformat(chat_response))
        if (chat_response["choices"][0]["message"].get("content")):
            output = chat_response["choices"][0]["message"]["content"]

        
    except Exception as e:
        # Handle other exceptions
        output = f"{type(e).__name__} {e}"
        LOGGER.error(f"Unable to generate chat response: {type(e).__name__} {e}")

    return output



async def is_correct_python( code ):

    model =  os.getenv('FAST_LLM', "gpt-3.5-turbo")


    messages = [
                {"role": "system", "content": "you are expert in python return true or false only"},
                {"role": "user", "content": f"check if following code is valid python code:\n {code}"}
            ]
    
    try:

        chat_completion_kwargs = {
            "messages": messages,
            "model": model,

        }
        # Make the chat completion request and parse the response
        LOGGER.info(pprint.pformat(chat_completion_kwargs))
        chat_response = await chat_completion_request(**chat_completion_kwargs)


        LOGGER.info(pprint.pformat(chat_response))

        output = True if chat_response["choices"][0]["message"]["content"].lower() in ["true" , "yes" , "ok"] else False

        
    except Exception as e:
        # Handle other exceptions
        output = f"{type(e).__name__} {e}"
        LOGGER.error(f"Unable to generate chat response: {type(e).__name__} {e}")

    return output
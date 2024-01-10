# import os
# from langchain.text_splitter import (
#     RecursiveCharacterTextSplitter,
#     Language,
# )
# from langchain.document_loaders import TextLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import TextLoader
# from dotenv import load_dotenv
# import os


# # Load environment variables from .env file
# load_dotenv()

# embeddings = OpenAIEmbeddings()


# openai_api_key = os.getenv('OPENAI_API_KEY')


# # You can also see the separators used for a given language
# RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

# root_dir = '/home/stahlubuntu/coder_agent/ros2'

# docs = []
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for file in filenames:
#         if file.endswith('.py') and '/.venv/' not in dirpath:
#             try: 
#                 loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
#                 docs.extend(loader.load_and_split())
#             except Exception as e: 
#                 pass
# print(f'{len(docs)}')


# from langchain.text_splitter import CharacterTextSplitter

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)

# # Embed and store the texts
# # Supplying a persist_directory will store the embeddings on disk
# persist_directory = 'db_ros2'

# embedding = OpenAIEmbeddings()
# vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
# vectordb.persist()
# vectordb = None


import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv


def persist_codebase(folder_name):
    # Load environment variables from .env file
    load_dotenv()

    embeddings = OpenAIEmbeddings()

    openai_api_key = os.getenv('OPENAI_API_KEY')

    # You can also see the separators used for a given language
    RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

    root_dir = f'/home/stahlubuntu/coder_agent/{folder_name}'

    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.py') and '/.venv/' not in dirpath:
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e: 
                    pass
    print(f'{len(docs)}')

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = f'db_{folder_name}'

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
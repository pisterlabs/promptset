import os
from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores import Pinecone
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import tqdm
import time
from argparse import ArgumentParser


def init(notion_dir_name):
    """Initializes the environment variables and the paths."""
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # find your environment next to the api key in pinecone console
    pinecone_env = os.getenv("PINECONE_ENV")
    # get our OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # path to the directory containing the notion documents
    notion_dir="./notion_data/"+notion_dir_name

    return pinecone_api_key, pinecone_env, notion_dir

def load_notion_db(notion_dir):
    """Loads the notion database from the specified directory and splits the documents into chunks of 500 characters with 0 overlap."""
    loader = NotionDirectoryLoader(notion_dir)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    print(f"we've split the documents into {len(all_splits)} chunks of 500 characters with 0 overlap")

    return all_splits

def init_pinecone_index(index_name, pinecone_api_key, pinecone_env):
    """Initializes the pinecone index."""
    
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536 # 1536 is the dimensionality of the OpenAI model
    )

    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pinecone.Index(index_name)

    return index

def embed_splits_openai(all_splits, index_name):
    """Embeds the splits using the OpenAI da Vinci model."""
    
    embeddings = OpenAIEmbeddings()

    vectordb = Pinecone.from_documents(all_splits[:1], embeddings, index_name=index_name)

    for i in tqdm.tqdm(range(1, len(all_splits))):
        vectordb.add_documents(all_splits[:i])

    return

# run our main function
if __name__ == '__main__':
    # get arguments from the command line
    parser = ArgumentParser()
    parser.add_argument("-n", "--notion", dest="notion_dir_name", help="what notion directory do you want to embed", metavar="NOTION_DIR", default="support_runbook")
    args = parser.parse_args()
    notion_dir_name = args.notion_dir_name
    index_name = 'notion-db-chatbot'
    print("Ok let's go!")
    pinecone_api_key, pinecone_env, notion_dir = init(notion_dir_name)
    print("Split the documents into chunks of 500 characters with 0 overlap.")
    all_splits = load_notion_db(notion_dir)
    print("Initializing the pinecone index...")
    index = init_pinecone_index(index_name, pinecone_api_key, pinecone_env)
    print("we've created an index and here is it's description") 
    index.describe_index_stats()
    print("let's embed the splits into the index, this might take some time and will cost you $")
    embed_splits_openai(all_splits, index_name)
    print("... and we're done! here is the index description again")
    index.describe_index_stats()
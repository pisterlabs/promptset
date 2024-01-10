import os
import argparse
import openai
import re
from pathlib import Path
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, StorageContext, load_index_from_storage, KeywordTableIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.llms import OpenAI
# You must obtain an API key from OpenAI for use of this script:
# https://platform.openai.com/account/api-keys
#
# TODO Replace this with your API key!
DEFAULT_OPENAI_API_KEY = 'YOUR_OPENAI_KEY_HERE'

def sanitize_filename(filename):
    # Remove any non-alphanumeric characters (except for underscores and hyphens)
    return re.sub(r'[^\w\-_]', '', filename)

def main():
    parser = argparse.ArgumentParser(description='Create a VectorStoreIndex from a PDF and query it.')
    parser.add_argument('pdf', help='The target PDF file')
    parser.add_argument('query', help='The query to run on the VectorStoreIndex')
    parser.add_argument('--key', default=DEFAULT_OPENAI_API_KEY, help='Your OpenAI API key')

    args = parser.parse_args()

    api_key = args.key or DEFAULT_OPENAI_API_KEY
    if api_key == 'YOUR_OPENAI_KEY_HERE':
        print("You must replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key in the script, or provide it using the --key flag.")
        print("For example: python3 ask_your_document.py --key 'YOUR_OPENAI_API_KEY' 'document.pdf 'What is the title of this document?'")
        return

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    # define LLM
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    service_context = ServiceContext.from_defaults(llm=llm)

    try:
        PyMuPDFReader = download_loader("PyMuPDFReader")
        loader = PyMuPDFReader()
        documents = loader.load(file_path=Path(args.pdf), metadata=True)

        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        # index.storage_context.persist(persist_dir="./storage") # bugged in latest version :(

        query_engine = index.as_query_engine()

        result = query_engine.query(args.query)
        print(result)
        return result

    except openai.error.AuthenticationError:
        print("An error occurred while trying to authenticate with the OpenAI API. Please ensure you've provided a valid API key.")


if __name__ == "__main__":
    main()

import sys
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import openai

def generate_response(messages, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2
        # Add any other parameters you want to use (e.g., temperature, max_tokens, etc.)
    )

    return response.choices[0].text.strip()

def similarity_search(query):
    # Load PDF and split into texts
    # loader = PyPDFLoader("/workspace/Apecoin.pdf")
    loader = PyPDFLoader("/Users/anky/Documents/GitHub/Creativedestruction.xyz/Apecoin.pdf")
    # loader = PyPDFLoader("/GPTBot/DATASET - Apecoin.pdf")
    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print(f'You have {len(data)} document(s) in your data')
    print(f'There are {len(data[30].page_content)} characters in your document')

    # Note: If you're using PyPDFLoader then we'll be splitting for the 2nd time.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    print(f'Now you have {len(texts)} documents')

    # Change this to environmental variables
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Initialize Pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "ape"  # Ape Assistant

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    # Perform similarity search
    docs = docsearch.similarity_search(query)

    # Here's an example of the first document that was returned
    response = docs[0].page_content[:450]

    # Prepare messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a friend cum assistant who answers succinctly. You follow a particular format and answer accoding to the input you've been given. Your aim is to help the user with their questions about the Apecoin DAO. You are provided with the additional context about a query, which you'll need to then respond accordingly. Try to be as helpful as you can and answer within the realm of the data provided. "},
    ]

    # Add the user's query as a new message
    messages.append({"role": "user", "content": query})

    # Generate response using GPT-3.5-turbo
    gpt_response = generate_response(messages, OPENAI_API_KEY)

    response += gpt_response

    return response

if __name__ == "__main__":
    # Read user query from stdin
    query = sys.stdin.readline().strip()
    result = similarity_search(query)
    print(result)
    sys.stdout.flush()
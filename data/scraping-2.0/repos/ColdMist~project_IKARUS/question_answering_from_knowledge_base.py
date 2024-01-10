from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQA, RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from utils.helper_functions import *
import argparse

def main(args):
    # setup the openAI API key
    os.environ["OPENAI_API_KEY"] = args.api_key
    # Create a completion
    # setup_openAI()
    # llm = OpenAI()

    # initialize the embeddings using openAI ada text embedding library
    embeddings = OpenAIEmbeddings()

    # initialize and read the *.pdf object
    texts = process_all_pdfs(args.directory_path, preprocess_langchain=True)

    # initialize the FAISS document store using the preprocessed text and initialized embeddings
    docsearch = FAISS.from_texts(texts, embeddings)
    retriever = docsearch.as_retriever()
    # Create a conversation buffer memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0), retriever=retriever, memory=memory
    )

    chat_history = []
    while True:
        # define the question
        print("type your question")
        query = input("")
        result = qa({"question": query, "chat_history": chat_history})
        print("system: ", result["answer"])
        chat_history.append((query, result["answer"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a directory path.')
    parser.add_argument('--directory_path', type=str, help='A directory path.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key.')
    args = parser.parse_args()
    main(args)

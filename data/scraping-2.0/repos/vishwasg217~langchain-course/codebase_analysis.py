from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate
import streamlit as st
from streamlit_chat import message
from dotenv import dotenv_values
import os

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

model = ChatOpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

# disallowed_special=() is required to avoid Exception: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte from tiktoken for some repositories
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API, disallowed_special=())    

# index codebase
CODEBASE_PATH  = "./the-algorithm"

def load_text_from_dir(path: str, encoding: str = "utf-8") -> list:
    """
    loading all the text in each of the file in the codebase and then appending it to a list. 
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, filename), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    return docs


def split_text(docs: list) -> list:
    """
    Splitting the text into sentences and then appending it to a list.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    texts = text_splitter.split_documents(docs)
    return texts

def deeplake_database(activeloop_id: str, dataset_name: str, splitted_texts: list):
    """
    Creating a new database and adding the splitted texts to the database.
    """
    dataset_path = f"hub://{activeloop_id}/{dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    print('Accessed database at {}'.format(dataset_path))
    db.add_documents(splitted_texts)
    return db


def search(db):
    """
    The `as_retriever` method is used in the Langchain library to convert a document search object into a retriever. This method is useful when you want to use the document search object in a retrieval pipeline.

    The `search_kwargs` is a parameter used in the Langchain library when performing a search using a retriever. It is a dictionary that contains the arguments for the search method.
    """
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    return retriever

def conversation(retriever):
    """
    The `ConversationalRetrievalChain` is a part of the Langchain library that allows for a conversational interaction with the AI. By default, the AI's responses are prefixed with "AI", and the human's inputs are prefixed with "Human". However, these prefixes can be customized.
    Langchain QA: https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb
    """

    qa = ConversationalRetrievalChain.from_llm(model, retriever = retriever)
    chat_history = []
    print("Enter your query (type exit to exit the program): ")
    print("---------------------------------------------------")
    query = ""
    while(query!="exit"):
        query = input("Human: ")
        print("******")
        if query == "exit":
            break

        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
        print(f"AI: {result['answer']} \n")




        
if __name__ == "__main__":

    use_my_database = False

    if use_my_database:
        docs = load_text_from_dir(CODEBASE_PATH)
        texts = split_text(docs)
        db = deeplake_database("vishwasg217", "twitter_algo_codebase", texts)
        retriever = search(db)
        conversation(retriever)
    else:
        db = DeepLake(
            dataset_path="hub://davitbun/twitter-algorithm",
            read_only=True,
            embedding_function=embeddings,
        )

        retriever = search(db)
        conversation(retriever)



    



    


        






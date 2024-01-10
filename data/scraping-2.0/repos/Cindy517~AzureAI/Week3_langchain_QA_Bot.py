from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()
import fitz

result = fitz.open()

for pdf in ['./Arxiv/project/PLTR-Ownership.pdf', 
            './Arxiv/project/PLTR-8K.pdf', 
            #'C:/Users/yrui7/Documents/vscode/Arxiv/project/PLTR-10K.pdf',
            './Arxiv/project/PLTR-10Q.pdf',]:
    with fitz.open(pdf) as mfile:
        result.insert_pdf(mfile)
    
result.save("./Arxiv/project/result.pdf")


if __name__ == "__main__":
    embeddings=OpenAIEmbeddings(deployment=os.getenv("AZURE_AI_ADA_EMBEDDING_DEPLOYMENT_NAME"),
                                model=os.getenv("AZURE_AI_ADA_EMBEDDING_DEPLOYMENT_MODEL_NAME"),
                                openai_api_base=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                                openai_api_type="azure",
                                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                chunk_size=1000)
    #dataPath = "./data/documentation/"
    fileName = "./Arxiv/project/result.pdf"

    #use langchain PDF loader
    loader = PyPDFLoader(fileName)

    #split the document into chunks
    pages = loader.load_and_split()

    #use langchain PDF directory loader
    #pdf_folder_path = "C:/Users/yrui7/Documents/vscode/Arxiv/project/"
    #loader = PyPDFDirectoryLoader(pdf_folder_path)
    #pages = loader.load_and_split()

    #Use Langchain to create the embeddings using text-embedding-ada-002
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    #save the embeddings into FAISS vector store
    db.save_local("./dbs/documentation/faiss_index")


import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#load environment variables
load_dotenv()

def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])


def ask_question_with_context(qa, question, chat_history):
    query = "what is Azure OpenAI Service?"
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


if __name__ == "__main__":
    # Configure OpenAI API
  
    llm = AzureChatOpenAI(openai_api_version="2023-05-15",
                            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                            model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
                            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                            openai_api_base=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                            #openai_api_type="azure"
                            )
    
    
    
    embeddings=OpenAIEmbeddings(deployment= os.getenv("AZURE_AI_ADA_EMBEDDING_DEPLOYMENT_NAME"),
                                model=os.getenv("AZURE_AI_ADA_EMBEDDING_DEPLOYMENT_MODEL_NAME"),
                                openai_api_base=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                                openai_api_type="azure",
                                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                chunk_size=1)


    # Initialize gpt-35-turbo and our embedding model
    #load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    #use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    #retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)


    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)
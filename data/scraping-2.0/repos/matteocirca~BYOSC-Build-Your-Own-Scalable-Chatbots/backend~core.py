import os
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from backend.llm import CustomLLM

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool

import hopsworks


# TODO: update db every 24h. Put VectorDB somewhere else than local?
def get_faiss_vectordb(inference_api_key, refresh=False):
    # initiate embeddings using HuggingFaceInferenceAPIEmbeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    # create a unique FAISS index path based on the input file's name
    faiss_index_path = "faiss_index_embeddings"

    try:
        if refresh:
            raise Exception("Refresh")
        db = FAISS.load_local(faiss_index_path, embeddings=embeddings)
        return db
    except:
        # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
        project = hopsworks.login()
        fs = project.get_feature_store()

        instruction_set = fs.get_feature_group(name="embeddings", version=1)
        df = instruction_set.read() # df is source, page, content

        # transform df in list of text
        # compose text like this for each row: "Source: source, Page: page, Content: content"
        documents = []
        for index, row in df.iterrows():
            documents.append(f"Source: {row['source']}, Page: {row['page']}, Content: {row['content']}")

        # split the loaded text into smaller chunks for processing
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=30,
        #     separators=["\n", "\n\n", "(?<=\. )", "", " "],
        # )
        # doc_chunked = text_splitter.split_documents(documents=documents)

        # print(documents)

        # create a FAISS vector database from the chunked documents and embeddings
        vectordb = FAISS.from_texts(documents, embeddings)
        
        # save the FAISS vector database locally using the generated index path
        vectordb.save_local(faiss_index_path)
        
        return vectordb

def run_llm(query, stop=None):
    # create an instance of the ChatOpenAI with specified settings
    # openai_llm = ChatOpenAI(temperature=0, verbose=True)

    # custom llm
    llm = CustomLLM(max_new_tokens=250, max_time=120.0)

    answer = llm._call(query, stop=stop)
    
    return answer


if __name__ == "__main__":
    db = get_faiss_vectordb(os.getenv('INFERENCE_API_KEY'))
    # db = FAISS.load_local("faiss_index_../slides/01_introduction", embeddings=HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv('INFERENCE_API_KEY'), model_name="sentence-transformers/all-MiniLM-l6-v2"))
    
    query = "How is the exam composed?"
    docs_and_scores = db.similarity_search_with_score(query)
    print(docs_and_scores[:5])


# importing all the necessary libraries
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,  OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI, HuggingFaceHub


if __name__ == "__main__":
    # data path
    data_path = './data'
    load_dotenv()

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # load the text files from the specified path
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    data = loader.load()

    # splitting the data to chunks to create a vectorstore
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents = text_splitter.split_documents(data)
    # using embedding from huggingface
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')

    #to use embedding from openai, uncomment this
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # create and save the vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore")
    # this will download the huggingface model if it hasnt been downloaded already
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
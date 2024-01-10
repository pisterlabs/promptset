import os
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub


from InstructorEmbedding import INSTRUCTOR




def generate_docs_from_directory(root_directory: str, target_extensions: tuple) -> list:
    documents = []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(target_extensions):
                file_path = os.path.join(dirpath, filename)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load_and_split())
                except Exception as e:
                    # print error and process.
                    print(
                        f"Error processing file {file_path}: {str(e)}. Passed")

    return documents


# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
#                                                       model_kwargs={"device": "cuda"})


def generate_embeddings_from_docs(documents,
                              chunk_size=1000,
                              chunk_overlap=0,
                              separators=["\n\n", "\n"],
                              save_directory=None,
                              embeddings=None):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    chunks = text_splitter.split_documents(documents)
    embeddings = embeddings if embeddings else HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
    save_directory = save_directory if save_directory else os.path.join(os.getcwd(), 'saved_embeddings')
    vector_store = DeepLake.from_documents(
        chunks, embeddings, dataset_path=save_directory)
    print(f"Vector store saved successfully at: {save_directory}")

    return vector_store



def generate_codebase_qa(model, retriever, temperature = 0.000001, max_length=256, return_source_documents=True):
    if model.startswith('gpt-'):
        llm = ChatOpenAI(model=model)
    else:
        llm = HuggingFaceHub(repo_id=model, model_kwargs={
                     "temperature": temperature, "max_length": max_length})
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=return_source_documents)
    


# codebase_qa = generate_codebase_qa(model = "databricks/dolly-v2-3b",
                                    
#                                        retriever=vector_store.as_retriever(),
#                                        return_source_documents=True)

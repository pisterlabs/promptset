from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_vector_store(text):
    """ 
    Returns vector store

    Parameters:
    text  (string) : takes the pdf text 

    Returns:
    Vector store  : returns the vector store 

    """
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text,embedding=embeddings)
    return vectorstore
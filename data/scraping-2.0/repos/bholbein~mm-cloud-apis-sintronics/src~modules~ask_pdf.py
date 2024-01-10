from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def get_answer(question: str):
    """
    This function creates an OpenAI language model instance based on the provided model and temperature values. 
    Then, it loads local embeddings from the given FAISS database path.

    Parameters:
    model (str): The model used by OpenAI.
    temperature (float): The temperature value used by OpenAI.
    db_path (str): The path of the FAISS database.

    Returns:
    None
    """
    llm = ChatOpenAI(temperature=0.0, model='gpt-4')
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("./vectorstore/faiss_index", embeddings)
    pdf_qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={
            "k": 4, "fetch_k": 8}),
        return_source_documents=True,
        verbose=False
    )
    result = pdf_qa({"query": question})
    return result

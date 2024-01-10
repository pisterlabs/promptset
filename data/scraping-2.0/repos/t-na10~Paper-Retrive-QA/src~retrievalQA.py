import os
import openai
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from src.utils import nougatOCR, text_splitter, prompt_qa

load_dotenv()


def RQA(
    pdf_path,
    question,
    constraint,
    model_name="gpt-3.5-turbo-16k",
):
    """_Execute RetrievalQA_

    Args:
        pdf_path (str): pdf file path.
        question (str): question.
        constraint (str): constraint.
        model_name (str): Defaults to "gpt-3.5-turbo-16k".

    Returns:
        str: result.
    """

    # Convert PDF to Markdown
    nougatOCR(pdf_path)
    pdf_name = pdf_path.split("/")[-1]
    mmd_name = pdf_name.replace(".pdf", ".mmd")
    mmd_path = f"./data/output/{mmd_name}"
    # Chunking
    texts = text_splitter(mmd_path)
    # Vector Store
    embedding = OpenAIEmbeddings(openai_api_type=os.environ['OPENAI_API_KEY'])
    db = FAISS.from_documents(texts, embedding)
    db.save_local("./db")
    # Initialise RetrievalQA Chain
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain_type_kwargs = {"prompt": prompt_qa(question, constraint)}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_key=os.environ['OPENAI_API_KEY'],
        ),
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
    )
    answer = qa.run(question)
    return answer

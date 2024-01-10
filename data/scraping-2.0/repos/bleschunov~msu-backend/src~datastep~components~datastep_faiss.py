import os
import pathlib
import shutil

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

def get_chain():
    # TODO: попробовать 3.5-instruct
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    template = """По данному тексту ответь на вопрос. Если для ответа на вопрос не хватает информации, напиши: Нет.

    Вопрос:
    {query}

    Текст:
    {text}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "text"]
    )

    return LLMChain(llm=llm, prompt=prompt)


def get_store_file_path(source_id: str) -> str:
    return f"{pathlib.Path(__file__).parent.resolve()}/../../../data/{source_id}/faiss"


def save_document(source_id: str, file_url: str):
    store_file_path = get_store_file_path(source_id)
    if os.path.isdir(store_file_path):
        return

    loader = PyPDFLoader(file_url)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    store_file_path = get_store_file_path(source_id)
    faiss_index.save_local(store_file_path)


def search(source_id: str, query: str):
    store_file_path = get_store_file_path(source_id)
    faiss_index = FAISS.load_local(
        store_file_path,
        OpenAIEmbeddings()
    )
    doc = faiss_index.similarity_search(query, k=1)
    return doc[0]


def query(source_id: str, query: str):
    chain = get_chain()
    doc = search(source_id, query)
    response = chain.run(
        query=query,
        text=doc.page_content
    )
    return doc.metadata["page"], response


def delete_document(source_id: str):
    store_file_path = get_store_file_path(source_id)
    shutil.rmtree(store_file_path)


if __name__ == "__main__":
    save_document(
        "Dog23012023_BI_3D_ispr_prava",
        "https://jkhlwowgrekoqgvfruhq.supabase.co/storage/v1/object/public/files/Dog23012023_BI_3D_ispr_prava.pdf"
    )
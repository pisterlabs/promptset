from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings

from discontinuity_api.vector import get_faiss_vector_db, add_document


def website_md_loading():
    loader = UnstructuredMarkdownLoader(
        file_path="./storage/website.md",
        metadata={"url": "https://discontinuity.ai/"},
    )
    documents = loader.load()

    db = get_faiss_vector_db(table_name="discontinuity", embeddings=OpenAIEmbeddings())

    add_document(index=db, documents=documents)

    db.save_local(folder_path="./storage/discontinuity")

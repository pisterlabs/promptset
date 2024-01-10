from fire import Fire
from RAG.utils import load_txt_as_list
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def index_and_store(
        references_file,
        output_index_file,
        embedding_hf_model='sentence-transformers/all-mpnet-base-v2',
):
    urls = load_txt_as_list(references_file)
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    # Converts HTML to plain text
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=33)
    chunked_documents = text_splitter.split_documents(docs_transformed)

    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name=embedding_hf_model))
    db.save_local(output_index_file)


if __name__ == '__main__':
    # Fire(index_and_store)
    index_and_store('static_urls/test_wiki.txt', output_index_file="vector_ds/test_wiki.faiss")

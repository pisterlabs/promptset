from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


def get_docs_list(file_path: str, separator='\n', chunk_size=39000, chunk_overlap=0) -> list[Document]:
    document = TextLoader(file_path, encoding='utf-8').load()
    split_docs = (CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                  .split_documents(document))
    return split_docs

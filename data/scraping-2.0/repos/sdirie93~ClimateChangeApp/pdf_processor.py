from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # Extract PDFs present in data folder
    )
    docs = loader.load()    # PDF data saved in variable
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

if __name__ == "__main__":
    doc_preprocessing()
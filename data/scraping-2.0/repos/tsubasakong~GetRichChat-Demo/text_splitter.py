from azure_file_loader import azureLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter():
    data = azureLoader()

    print("data", data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # print(texts)

    return texts 



if __name__ == "__main__":
    get_text_splitter()


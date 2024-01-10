from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
import weaviate


def add_document(name):
    loader = UnstructuredPDFLoader(name)

    data = loader.load()

    print(f'There are {len(data[0].page_content)} characters in the document. \n')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)


    print(f'There are {len(texts)} chunked documents now.')
    print(f'The first one contains \n\n {texts[0]} \n')

    # docker-compose up -d to run, docker-compose down to stop
    client = weaviate.Client("http://localhost:8080")
    vectorstore = Weaviate(client, "Document", "text")
    vectorstore.add_documents(texts) 
    print('Documents added to database.')


if __name__ == "__main__":
    doc_name = input('Which file do you want to add to the database? \n')
    add_document(doc_name)

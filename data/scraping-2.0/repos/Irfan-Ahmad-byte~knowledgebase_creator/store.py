from langchain.vectorstores import Chroma
from langchain import embeddings


#  uncomment for linux
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def store_docs(docs, embedding_function: embeddings):
    '''
        function to store documents

        parameters:
            docs: list of documents to store
            embeddings: embeddings to apply

        returns:
            vector_store: vector store object
    '''

    print('Storing documents...')

    #store documents
    db = Chroma.from_documents(docs, embedding_function)

    print('Documents stored successfully!')

    return db
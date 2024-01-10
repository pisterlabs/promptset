from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import document_loaders


def process_docs(docs: document_loaders):
    '''
        function to split text into sentences and words

        parameters:
            docs: list of document objects

        returns:
            texts: list of text objects
    '''

    #load text splitter
    text_splitter = RecursiveCharacterTextSplitter(
                                                        chunk_size=700,
                                                    )
    #split documents
    print('Splitting documents...', docs[0].metadata)
    texts = text_splitter.split_documents(docs)

    return texts



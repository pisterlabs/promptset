import os
import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--api_key',
        type=str,
        help="openai_key",
    )
    
    config = p.parse_args()

    return config

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def query_contra(text, persist_directory, embedding):
    # persist_directory = './DB/contrabass'
    
    vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(max_tokens=-1),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

    query = text
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
    print('')

def main(config):
    os.environ["OPENAI_API_KEY"] = config.api_key
    # test3
    loader = TextLoader('./콘트라베이스_표준제안서.txt')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    persist_directory = './db/contrabass'
    embedding = OpenAIEmbeddings()
    
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):        
        pass
    else:
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=persist_directory)
        vectordb.persist()
        
    vectordb = None

    # while True:
    #     text = input('Question: ')
    #     print('Answering to your question.....')
    #
    #     query_contra(text, persist_directory, embedding)

    text = "콘트라베이스의 핵심 원리가 뭔데?"
    print('Answering to your question.....')

    query_contra(text, persist_directory, embedding)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
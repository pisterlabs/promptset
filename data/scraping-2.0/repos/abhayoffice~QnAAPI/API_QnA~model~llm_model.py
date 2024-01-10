from langchain.llms import CTransformers

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import os


def my_llm_model(documents):
    config = {'max_new_tokens': 1024,
              'repetition_penalty': 1.1,
              'temperature': 0.1,
              'top_k': 50,
              'top_p': 0.9,
              'context_length': 2048,
              'stream': True,
              'threads': int((os.cpu_count()))
              }

    llm = CTransformers(model='TheBloke/zephyr-7B-alpha-GGUF', model_file='zephyr-7b-alpha.Q5_K_S.gguf',
                        model_type="mistral", lib="avx2", config=config, callbacks=[StreamingStdOutCallbackHandler()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={'device': 'cpu'},
    )

    from langchain.vectorstores import FAISS

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss")
    db = FAISS.load_local("faiss", embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 2})
    template = ''' Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context={context}
    Question={question}

    only return the helpful answer below and nothing else.

    Helpful answer:
    '''
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        chain_type="stuff",
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})

    return chain




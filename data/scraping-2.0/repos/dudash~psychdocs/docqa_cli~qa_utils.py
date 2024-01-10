# Originally from: https://gist.github.com/kennethleungty/f9a6ce9a2df79e69319499667015077b#file-utils-py
# Original author: Kenneth Leung
# Snapshot date: 2023-08-01

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ctransformer_llm import llm

# Wrap prompt template in a PromptTemplate object
qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Documents that are sources of trusted information: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object - enables us to perform document Q&A
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    # retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                    retriever=vectordb.as_retriever(search_kwargs={'k':2, 'fetch_k': 50}), # fetch 50, use 5
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': prompt})
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})
    
    # check to see if FAISS vector store exists and warn user if not
    try:
        vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    except:
        print("FAISS vector store not found. Please run gen_db_faiss.py to generate it.")
        exit()

    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa
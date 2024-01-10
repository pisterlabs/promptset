import warnings
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader  #for textfiles
from langchain.text_splitter import CharacterTextSplitter #text splitter
from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader  #load urls into docoument-loader

'''
import requests
url2 = "https://github.com/fabiomatricardi/cdQnA/raw/main/KS-all-info_rev1.txt"
res = requests.get(url2)
with open("KS-all-info_rev1.txt", "w") as f:
    f.write(res.text)
'''

import os

def generate_response(txt_file, query):

    os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'xxx'

    # Document Loader
    from langchain.document_loaders import TextLoader
    loader = TextLoader(txt_file)
    documents = loader.load()
    import textwrap
    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')
        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    # Text Splitter
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()

    #Create the vectorized db
    # Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
    from langchain.vectorstores import FAISS
    db = FAISS.from_documents(docs, embeddings)


    from langchain.chains.question_answering import load_qa_chain
    from langchain import HuggingFaceHub

    llm6 = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-783M", model_kwargs={"temperature":0, "max_length":512})
    chain = load_qa_chain(llm6, chain_type="stuff")
    #our questions
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    return response

if __name__ == '__main__':
    generate_response('./sample_pdf/KS-all-info_rev1.txt', 'What is Hierarchy 4.0?')
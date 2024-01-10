"""
記事情報ベクトル化エリア
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import pickle
import os
import re
doc_dir = '../../storage/kumeharadocuments'
documents = []
out_doc = ''
in_doc = ''
def save(text,path):
    with open(path, "w") as file:
        file.write(text)
def read(path):
    with open(path, "r") as file:
        return file.read()
    

    
text_splitter = RecursiveCharacterTextSplitter(
# Set a really small chunk size, just to show.
chunk_size = 200,
chunk_overlap  = 50,
length_function = len,
)   
def main():
    docs = []
    meta_datas = []
    for filename in os.listdir(doc_dir):
        filepath = os.path.join(f'{doc_dir}', filename)
        loader = TextLoader(filepath)
        file = loader.load()[0]
        doc = file.page_content
        meta_data = file.metadata
        docs.append(doc)
        meta_data['source'] = meta_data['source'].replace('./kumeharadocuments/','').replace('.txt','')
        meta_datas.append(meta_data)

    documents = text_splitter.create_documents(docs,meta_datas)
    # save(out_doc,f'./liny-manual-chatbot/除外段落.txt')
    # save(in_doc,f'./liny-manual-chatbot/適用段落.txt')

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)


    # Save vectorstore
    with open("../../storage/kumeharadocuments/vectorstore_kume__200_50.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == '__main__':
    main()
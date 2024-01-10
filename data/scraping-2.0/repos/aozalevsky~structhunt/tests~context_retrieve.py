import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

import re
import requests
import xml.etree.ElementTree as ET

from fragment import Fragment
from VectorDatabase import Latern



# OpenAI Setup
OPEN_API_KEY = "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"
# openai.api_key = os.getenv(openai_api_key)
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY

def getPmcPaper(pmcid):
    """
    """
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    req = requests.get(url)
    res = req.text
    return res

def extractMethodsFromPmcPaper(paper):
    """
    """
    tree = ET.fromstring(paper)
    mtext = []
    for sec in tree.iter('sec'):
        for title in sec.iter('title'):
            if isinstance(title.text, str):
                if re.search('methods', title.text, re.IGNORECASE):
                    mtext.extend(list(sec.itertext()))
    return " ".join(mtext)

def preprocess(input_text):
    """
    """
    processed_data = input_text.replace("\n","")
    return processed_data

def get_embeddings(fname):
    """
    """
    loader = TextLoader(fname)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator = ".",chunk_size = 1000, chunk_overlap=0)
    
    docs = text_splitter.split_documents(documents)
    
    emb = OpenAIEmbeddings()
    input_texts = [d.page_content for d in docs]

    input_embeddings = emb.embed_documents(input_texts)
    text_embeddings = list(zip(input_texts, input_embeddings))

    return text_embeddings, emb

def saveFassIndex(fname, sname, ):
    """
    """
    txt_embs, emb = get_embeddings(docs)
    faissIndex = FAISS.from_embeddings(text_embeddings=txt_embs, embedding=emb)
    faissIndex.save_local(sname)
    # faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    # faissIndex.save_local("input_doc")

def Query(input_query, faiss_obj):
    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=OPEN_API_KEY,
            temperature=0, model_name="gpt-3.5-turbo", max_tokens=50
        ),
        chain_type="stuff",
        retriever=faiss_obj.as_retriever(search_type="similarity", search_kwargs={"k":1})
    ) 
    template = """ {query}? """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )
    print(chatbot.run(
        prompt.format(query=input_query)
    ))


def main():
    text = getPmcPaper(pmcid)
    
    methods_text = preprocess(extractMethodsFromPmcPaper(text))
    fname = 'input_file.txt'
    sname = 'input_doc'
    with open(fname, 'w') as file:
        file.write(methods_text)
    # print(methods_text)
    txt_embs, emb = get_embeddings(fname) 
    
    fragments = []
    for txt, embs in txt_embs:
        fragment = Fragment(pmcid, 'methods', txt, embs)
        fragments.append(fragment)
        
    latern = Latern()
    latern.insertEmbeddings(fragments)
    
    # retreieve. PMC
    faissIndex = FAISS.from_embeddings(text_embeddings=txt_embs, embedding=emb)
    inp_query = "Does the paper report a new structure of a biomolecule or biomolecular complex modeled using experimental data"
    Query(inp_query, faissIndex)

if __name__ == '__main__':  
    main()
import pandas as pd
import streamlit as st
import os
import json
import csv
import tiktoken
import numpy as np
import time

import re
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup

#sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
	MarkdownTextSplitter,
	PythonCodeTextSplitter,
	RecursiveCharacterTextSplitter)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAIChat

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATA_FRACTION, EMBEDDING_MODELS, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR

from bardapi import Bard
import openai

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#question	jeopardy_answer	kb-gpt35_answer	kb-gpt4_answer	kb-40b_answer	kb-llama2-13b_answer	kb-llama2-13b_templated_answer	kb-llama2-70b_answer (4 bit)	kb-llama2-70b_answer (8 bit)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# Will download the model the first time it runs
embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    cache_folder="../models/sentencetransformers",
)

# get embedding for one sentence
def get_embedding(sentence):
    try:
        return embedding_function.embed_documents([sentence])[0]
    except Exception as e:
        print(e)
        return np.zeros(384)


# make sure load_dotenv is run from main app file first
openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

#bard = Bard(token=os.getenv('BARD_API_KEY'))


def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "uploaded_file": None,
        "generation_model": MODELS[0],
        "general_context": "",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": []
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_new_top_pos(c, prev_top_pos=None):
    try:
        return int(re.findall("top:\d+px",c.attrs['style'])[0][4:-2])
    except Exception as e:
        print(e)
        return prev_top_pos

def parse_pdf_document(this_pdf):
    """ Function to read pdf and split the content into a list of documents"""
    loader = PDFMinerPDFasHTMLLoader(this_pdf)
    data = loader.load()[0]   # entire PDF is loaded as a single Document

    soup = BeautifulSoup(data.page_content,'html.parser')
    content = soup.find_all('div')
    
    # cur_fs = None
    cur_text = ''
    last_top_pos = 0
    new_page = True    

    metadata={}
    metadata.update(data.metadata)

    docs = []   # first collect all snippets that have the same font size
    #if top of page skip and continue
    for idx, c in enumerate(content):
        new_top_pos = get_new_top_pos(c, prev_top_pos=last_top_pos)
        if c.text=='Transcribed by readthatpodcast.com \n':
            new_page = True
            continue

        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            print(fs)
            continue
        # fs = int(fs[0])
        # if not cur_fs:
        #     cur_fs = fs
        if not last_top_pos:
            last_top_pos = new_top_pos

        #check if not 2 line spaces or if new page is a continuation of previous line
        if new_top_pos<last_top_pos+30 or (new_page and not c.text[0].isupper()):
            cur_text += c.text
        elif new_page:
            docs.append(Document(page_content=cur_text,metadata=metadata.copy()))
            # cur_fs = fs
            cur_text = c.text
        # elif not c.text.endswith(".  \n") and len(c.text)<50: #if next line is not a full line, append to current line
        #     cur_text += c.text
        else:        
            docs.append(Document(page_content=cur_text,metadata=metadata.copy()))
            # cur_fs = fs
            cur_text = c.text

        last_top_pos = new_top_pos
        new_page = False
        
    if cur_text!='':    
        docs.append(Document(page_content=cur_text,metadata=metadata.copy()))

    section="Introduction"
    new_section=False
    final_docs=[]
    doc_idx=0
    #combine document sections based on provided timestamps
    for idx, doc in enumerate(docs):
        #check if new section / if it was a timestamp
        timestamp=re.search("\d+:\d+:\d+",doc.page_content)
        if not timestamp:
            timestamp=re.search("\d+:\d+",doc.page_content)
        if idx==0:
            doc.metadata.update({'section':section,'doc_idx':doc_idx})
            final_docs.append(doc)
            doc_idx+=1
        elif timestamp and timestamp.start()==0 and not new_section:
            section=doc.page_content   
            new_section=True
            if idx<len(docs)-1:
                #get the last sentence from the previous content page
                last_sent=docs[idx-1].page_content.split(".")[-1]
                if len(last_sent)<10:
                    last_sent=docs[idx-1].page_content

                # CHANGE THIS TO ITERATE OVER SENTENCES INSTEAD OF JUST LOOK AT THE FIRST SENTENCE
                next_sent=docs[idx+1].page_content.split(".")[0]
                if next_sent[0].islower() and len(next_sent)<50:
                    final_docs[-1].page_content=final_docs[-1].page_content + next_sent + "."
                    docs[idx+1].page_content=".".join(docs[idx+1].page_content.split(".")[1:])#remove the first sentence from the next document
                elif len(next_sent)<len(docs[idx+1].page_content):
                    this_emb=get_embedding(section)
                    last_emb=get_embedding(last_sent)
                    next_emb=get_embedding(next_sent)
                    #if the next sentence is more similar to the previous sentence than the current section, then combine
                    if cosine_similarity([this_emb],[next_emb])[0][0] <cosine_similarity([last_emb],[next_emb])[0][0]:
                        final_docs[-1].page_content=final_docs[-1].page_content + next_sent + "."
                        docs[idx+1].page_content=".".join(docs[idx+1].page_content.split(".")[1:]) #remove the first sentence from the next document
        else:
            # metadata=doc.metadata
            doc.metadata.update({'section':section,'doc_idx':doc_idx})
            if new_section:
                doc.page_content=section + "\n" + doc.page_content
                new_section=False     
            # doc.metadata=metadata
            final_docs.append(doc)
            doc_idx+=1

    return final_docs


# This is a dummy function to simulate generating responses.
def generate_responses(prompt, model, template="", temperature=0):
    response = "No model selected"

    if model != "None":
        st.session_state["generation_models"] = model

        if model.startswith("Google"):
            this_answer = bard.get_answer(prompt)
            response = this_answer['content']
        elif model.startswith("OpenAI: "):
            # try to call openai and if it fails wait 5 seconds and try again
            try:
                response_full = openai.Completion.create( model=model[8:],  messages=[{"role": "user", "content": prompt }], temperature=temperature)
            except:
                st.warning("OpenAI API call failed. Waiting 5 seconds and trying again.")
                time.sleep(5)
                response_full = openai.ChatCompletion.create( model=model[8:],   messages=[{"role": "user", "content": prompt }], temperature=temperature)
            response = response_full['choices'][0]['message']['content']

    return response



def split_json_doc_with_header(doc):
    """Separate header on line one from json doc and split json dict by keys"""
    try:
        header = doc.page_content.split("\n")[0]
        #print(doc.page_content[len(header)+1:])
        json_dict = json.loads(doc.page_content[len(header)+1:])
        doc_list = []
        for key in json_dict.keys():
            doc_list.append(Document(page_content=header+'Data for ' +str(key)+ ':\n'+str(json_dict[key]), metadata=doc.metadata))
        return doc_list
    except Exception as e:
        print(e)
        print("Unable to split " + doc.metadata['source'])
        return [doc]




def create_knowledge_base(docs):
    """Create knowledge base for chatbot."""

    print(f"Loading {PROCESSED_DOCUMENTS_DIR}")
    docs_orig = docs
        
    print(f"Splitting {len(docs_orig)} documents")
    # docs = []
    # for doc in docs_orig:
    #     print(doc)
    #     num_tokens = num_tokens_from_string(doc.page_content)
    #     if  num_tokens > int(.1*MAX_TOKENS):
    #         doc_list = split_json_doc_with_header(doc)
    #         docs.extend(doc_list)
    #     else:
    #         docs.append(doc)
    
    print(f"Created {len(docs)} documents")

    # Will download the model the first time it runs
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODELS[0],
        cache_folder="../models/sentencetransformers",
    )
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. You may want to increase the number of CPU's in your noteboook.
        """
    )
    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)  
    # Save the FAISS db 
    db.save_local("../data/faiss-db")

    print(f"FAISS VectorDB has {db.index.ntotal} documents")
    

def generate_kb_response(prompt, model, template=None):

    data_dict = {}
    data_dict['prompt'] = prompt
    data_dict['chat_history'] = []

    if model.startswith("OpenAI: "):
        llm = OpenAIChat(model=model[8:], max_tokens=3000, temperature=TEMPERATURE)
    else:
        return "Please select an OpenAI model."

    # Will download the model the first time it runs
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODELS[0],
        cache_folder="../models/sentencetransformers",
    )
    db = FAISS.load_local("../data/faiss-db", embedding_function)

    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,return_source_documents=True) #, return_source_documents=True

    # prompt_template = """
    # ### System:
    # {context}

    # ### User:
    # {question}

    # ### Assistant:
    # """
    # PROMPT = PromptTemplate(
    #     template=prompt_templatjosleee, input_variables=["context", "question"]
    # )
    # chain_type_kwargs = {"prompt": PROMPT}
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    # query = data_dict['prompt']
    # return qa.run(query)

    response = chain(inputs={'question':data_dict['prompt'], 'chat_history':data_dict['chat_history']})
    print(response)
    

    return response['answer']


import streamlit as st
from code_editor import code_editor

from genai.credentials import Credentials
from genai.schemas import GenerateParams
from genai.model import Model

import tempfile
import pathlib
import re

from unstructured.partition.auto import partition
import nltk
import ssl

import os
from dotenv import load_dotenv
from lxml import html
from pydantic import BaseModel
from typing import Any, Optional
from unstructured.partition.pdf import partition_pdf

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma

import uuid

load_dotenv()

api_key = st.secrets["GENAI_KEY"]
api_endpoint = st.secrets["GENAI_API"]

api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

creds = Credentials(api_key,api_endpoint)

params = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=1000,
    min_new_tokens=1,
    # stream=True,
    top_k=50,
    top_p=1
)

model = Model(model="meta-llama/llama-2-70b-chat",credentials=creds, params=params)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=HuggingFaceEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

def fillvectordb(table_elements,text_elements):
    table_summaries, text_summaries = buildsummary(table_elements,text_elements)
    
    # Add texts
    texts = [i.text for i in text_elements]
    doc_ids = [str(uuid.uuid4()) for _ in texts]

    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    tables = [i.text for i in table_elements]
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

def buildsummary(table_elements,text_elements):
    summary_prompt_text= \
    """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: \
    {content} \
    summary:"""

    #async
    tables = [i.text for i in table_elements]
    total = len(tables)

    table_summaries = []
    table_prompts = [summary_prompt_text.format(content=table) for table in tables]

    i = 0
    for result in model.generate_async(table_prompts):
        i += 1
        print("[Progress {:.2f}]".format(i/total*100.0))
        print("\t {}".format(result.generated_text))
        table_summaries += [result.generated_text]

    texts = [i.text for i in text_elements]
    total = len(texts)

    text_summaries = []
    text_prompts = [summary_prompt_text.format(content=text) for text in texts]

    i = 0
    for result in model.generate_async(text_prompts):
        i += 1
        print("[Progress {:.2f}]".format(i/total*100.0))
        print("\t {}".format(result.generated_text))
        text_summaries += [result.generated_text]

    return table_summaries, text_summaries

def buildquestion(table_elements,text_elements):
    question_prompt_text = """Generate a list of 3 hypothetical questions that the below document could be used to answer: \
    {content} \
    hypothetical questions:"""

    #async
    tables = [i.text for i in table_elements]
    total = len(tables)

    table_questions = []
    table_prompts = [question_prompt_text.format(content=table) for table in tables]

    i = 0
    for result in model.generate_async(table_prompts):
        i += 1
        print("[Progress {:.2f}]".format(i/total*100.0))
        print("\t {}".format(result.generated_text))
        table_questions += [result.generated_text]

    texts = [i.text for i in text_elements]
    total = len(texts)

    text_questions = []
    text_prompts = [question_prompt_text.format(content=text) for text in texts]

    i = 0
    for result in model.generate_async(text_prompts):
        i += 1
        print("[Progress {:.2f}]".format(i/total*100.0))
        print("\t {}".format(result.generated_text))
        text_questions += [result.generated_text]

    return table_questions, text_questions

def ingestpdf(pdffile):
    st.write('start ingest')
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Get elements
    st.write('before partition')
    raw_pdf_elements = partition_pdf(filename=pdffile,
                                    extract_images_in_pdf=False,
                                    infer_table_structure=True, 
                                    chunking_strategy="by_title",
                                    max_characters=4000, 
                                    new_after_n_chars=3800, 
                                    combine_text_under_n_chars=2000,
                                    image_output_dir_path='.')
    st.write('done partition')
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())
    category_counts

    class Element(BaseModel):
        type: str
        text: Any


    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))

    fillvectordb(table_elements,text_elements)

def queryvectordb(retriever, question):
    return retriever.get_relevant_documents(
        question
    )

def buildpromptforquery(question,informations):
    return f"""
    answer the question in 5 sentences base on the informations:
    informations:
    {{informations}}
    question:
    {question}
    answer in point form:"""

def querypdf(question):
    question = "how to handle if one engine of flight not work?"

    informations = queryvectordb(retriever,question)

    prompt = buildpromptforquery(informations,question)

    prompts = [buildpromptforquery(informations,question)]
    answer = ""
    for response in model.generate_async(prompts,ordered=True):
        answer += response.generated_text
    return answer

def uploadpdf(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    if uploaded_file is not None:
        st.write(f"filename:{uploaded_file.name}")
        fullpath = os.path.join(pathlib.Path(temp_dir.name),uploaded_file.name)
        st.write(f"fullpath:{fullpath}")
        with open(os.path.join(pathlib.Path(temp_dir.name),uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

        if fullpath.lower().endswith('.pdf'):
            ingestpdf(fullpath)

# st.set_page_config(layout="wide")
st.header("Document RAG powered by watsonx")

with st.sidebar:
    st.title("Document RAG")
    uploadpdf(st.file_uploader(label="upload a document",type=['pdf']))

with st.chat_message("system"):
    st.write("please input your query")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("your query"):
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    answer = querypdf(query)

    st.session_state.messages.append({"role": "agent", "content": answer}) 

    with st.chat_message("agent"):
        st.markdown(answer)
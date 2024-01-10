# langchain_helper.py

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


import numpy as np
import io


import os
import tempfile
import whisper

import librosa
import numpy as np

from langchain.document_loaders import TextLoader

import tempfile
import shutil


load_dotenv()


# ... (rest of the code above)

def create_vector_db_from_memos2(memo_files, api_key, st):

    model = whisper.load_model("base")
    docs = []
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    total_files = len(memo_files)
    
    # Progress bar initialization
    progress_bar = st.progress(0)
    status = st.empty()
    status.text("Transcribing memos...")

    for idx, uploaded_file in enumerate(memo_files):

        # Transcribe audio file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            shutil.copyfileobj(uploaded_file, temp_file)
            temp_file_path = temp_file.name

        result = model.transcribe(temp_file_path)
        
        # Load transcription text into docs
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(result["text"])
            f.flush()
            
            loader = TextLoader(f.name)
            docs.extend(loader.load())

        os.remove(temp_file_path)
        
        progress_value = (idx+1) / total_files
        progress_bar.progress(progress_value)

    # Split docs and create database
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    db = FAISS.from_documents(docs, embeddings)

    return db



def create_document(text):
    return {"page_content": text}



def get_response_from_query(db, query, openai_api_key, st):
    # text-danvinci can handle 4097 tokens

    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003", openai_api_key=openai_api_key)



    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about a user's transcribed voice memos. 
        
        Answer the following question: {question}
        By searching the following memos: {docs}

        Be extremely descriptive. Even if there are only brief mentions of the answer in the memos, that is fine, still bring it up to the user.

        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    print("Query:", query)
    print("Docs:", docs_page_content)

    st.text("Asking your question...")
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    st.text("Response received!")

    return response
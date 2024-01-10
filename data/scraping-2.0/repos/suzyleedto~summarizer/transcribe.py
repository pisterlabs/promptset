

import streamlit as st
import streamlit_ext as ste
import openai
import os
import io
import csv
from pydub import AudioSegment 
import pysqlite3
import sys 

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from streamlit_chat import message
from langchain import PromptTemplate
import tempfile


sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
openai.organization = "org-ydtCQcRROzj3YuGKoh4NtXEV"
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
#openai.api_key = os.environ['OPENAI_API_KEY'] 
#openai_api_key = os.environ['OPENAI_API_KEY'] 
llm = OpenAI(temperature=0.1)
openai.Model.list()


def transcribe_audio(audio_segment):
    chunk_length_ms = 20 * 60 * 1000
    # split audio file into 10 second chunks
    chunks = [audio_segment[i:i+chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
    transcript_all  = "Interview Transcription:\n"
    for i, chunk in enumerate(chunks):
        # Export the chunk to a temporary WAV file
        temp_file = f"temp_chunk_{i}.mp3"
        chunk.export(temp_file, format="mp3")
        st.write(f"sending to openai chunk {i+1} out of {len(chunks)}")
        audio_file= open(temp_file, "rb")
        transcript = openai.Audio.transcribe("whisper-1", file = audio_file,prompt = "This audio contains an interview with Filipino and English on the topic of usage of data.")
        transcript_all =  transcript_all+transcript.text   
    return transcript_all


def summarize(source_text):
    # If the 'Summarize' button is clicked
    try:
        with st.spinner('Please wait...'):
             # Split the source text
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(source_text)
            # Create Document objects for the texts (max 3 pages)
            docs = [Document(page_content=t) for t in texts[:10]]
            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(temperature=0, openai_api_key=openai.api_key, model = 'gpt-3.5-turbo-16k-0613')
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)
            st.success(summary)
    except Exception as e:
        st.exception(f"An error occurred: {e}")


def write_file(text, filepath):
   with open(filepath, 'w') as f:
       f.write(text)
       print(f"Data has been written to {filepath}")
        


def qa_file(filepath):
    
    if 'chain' not in st.session_state:
        loader = TextLoader(file_path=filepath)
        data = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100,separator="?")
        
        print("splitting")
        texts = text_splitter.split_documents(data)
            
            
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k":5})
        print("splitting")
        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.1,model = 'gpt-3.5-turbo-16k', openai_api_key=openai_api_key),
                                                                                retriever=retriever)
    
        st.session_state['chain'] = chain 
    def conversational_chat(query):
        print("sending q")  
        chain =  st.session_state['chain']
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
                
        return result["answer"]
            
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about the interview 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
                
        #container for the chat history
    response_container = st.container()
        #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=None)
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                    
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i),avatar_style="initials", seed = "DTO")


def main():
    transcript = ""
    st.title("DTO Interview Transcription and Summarizer App")
    option = st.sidebar.selectbox('What would you like to do?',('Upload an Audio File', 'Upload a Text File'))
    if option == 'Upload an Audio File':
       uploaded_file = st.sidebar.file_uploader("Upload an audio file for transcription", type=["wav", "mp3", "flac", "m4a"])
       if uploaded_file is not None:
            st.write(uploaded_file)
            try:
                    # Read the uploaded file
                if 'transcript' not in st.session_state:
                    with st.spinner("Transcribing file..."):
                        st.write("Reading audio file...")
                        audio_data = uploaded_file.read()
                        # Create an AudioSegment object from the file data
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                        st.write("Splitting audio file...")
                        transcript = transcribe_audio(audio_segment)
                        st.session_state.transcript = transcript
                    st.success('Transcript completed!!', icon="✅")
                    write_file(transcript, "output.txt")
                    with st.expander("See Transcript"):
                        st.text_area("Transcript", transcript, height=200)
                    with open('output.txt') as f:
                        ste.download_button('Download txt file for future use', data = f, file_name = "transcript.txt")  # Defaults to 'text/plain'
                    st.info("To summarize, select Upload a Text File from the dropdown", icon = "👈")
            except Exception as e :
                st.exception(f"An error occurred: {e}")
    elif option == 'Upload a Text File':        
        uploaded_txt_file = st.sidebar.file_uploader("Upload a text file with a transcript", type=["txt", "doc","docx"])         
        if uploaded_txt_file is not None :
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_txt_file.getvalue())
                tmp_file_path = tmp_file.name
                qa_file(tmp_file_path)




if __name__ == "__main__":
 
    main()
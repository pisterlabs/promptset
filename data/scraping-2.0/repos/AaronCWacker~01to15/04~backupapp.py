import streamlit as st
import openai
import os
import base64
import glob
import json
import mistune
import pytz
import math
import requests
import time
import re
import textract
import zipfile  # New import for zipping files


from datetime import datetime
from openai import ChatCompletion
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from collections import deque
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from templates import css, bot_template, user_template

# page config and sidebar declares up front allow all other functions to see global class variables
st.set_page_config(page_title="GPT Streamlit Document Reasoner", layout="wide")
should_save = st.sidebar.checkbox("💾 Save", value=True)

def generate_filename_old(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")  # Date and time DD-HHMM
    safe_prompt = "".join(x for x in prompt if x.isalnum())[:90]  # Limit file name size and trim whitespace
    return f"{safe_date_time}_{safe_prompt}.{file_type}"  # Return a safe file name

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        #st.write('Responses:')
        #st.write(chatResponse)
        filename = generate_filename(transcript, 'txt')
        #create_file(filename, transcript, chatResponse)
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return

    # Step 2: Extract base filename without extension
    base_filename, ext = os.path.splitext(filename)

    # Step 3: Check if the response contains Python code
    has_python_code = bool(re.search(r"```python([\s\S]*?)```", response))

    # Step 4: Write files based on type
    if ext in ['.txt', '.htm', '.md']:
        # Create Prompt file
        with open(f"{base_filename}-Prompt.txt", 'w') as file:
            file.write(prompt)
        
        # Create Response file
        with open(f"{base_filename}-Response.md", 'w') as file:
            file.write(response)

        # Create Code file if Python code is present
        if has_python_code:
            # Extract Python code from the response
            python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()

            with open(f"{base_filename}-Code.py", 'w') as file:
                file.write(python_code)


def create_file_old(filename, prompt, response, should_save=True):
    if not should_save:
        return
    if filename.endswith(".txt"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}\n{response}")
    elif filename.endswith(".htm"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}   {response}")
    elif filename.endswith(".md"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}\n\n{response}")
            
def truncate_document(document, length):
    return document[:length]
def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        try:
            data = file.read()
        except:
            st.write('')
            return file_path    
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
        
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []

    for chunk in openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
        temperature=0.5,
        stream=True  
    ):
        
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        
        content=chunk["choices"][0].get("delta",{}).get("content")
        
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                #result = result.replace("\n", "")        
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
        
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    # Check if the input is a string
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    # If it's not a string, assume it's a streamlit.UploadedFile object
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

from io import BytesIO
import re

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        # print the file extension
        st.write(f"File type extension: {file_extension}")

        # read the file according to its extension
        try:
            if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
                text += file.getvalue().decode('utf-8')
            elif file_extension.lower() == 'pdf':
                from PyPDF2 import PdfReader
                pdf = PdfReader(BytesIO(file.getvalue()))
                for page in range(len(pdf.pages)):
                    text += pdf.pages[page].extract_text() # new PyPDF2 syntax
        except Exception as e:
            st.write(f"Error processing file {file.name}: {e}")

    return text

def pdf2txt_old(pdf_docs):
    st.write(pdf_docs)
    for file in pdf_docs:
        mime_type = extract_mime_type(file)
        st.write(f"MIME type of file: {mime_type}")
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        # Save file output from PDF query results
        filename = generate_filename(user_question, 'txt')
        #create_file(filename, user_question, message.content)
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       
        #st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1  # Adding 1 to account for spaces
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))  # Append the final chunk
    return chunks

def create_zip_of_files(files):
    """
    Create a zip file from a list of files.
    """
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name


def get_zip_download_link(zip_file):
    """
    Generate a link to download the zip file.
    """
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

    
def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # File type for output, model choice
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))

    # Audio, transcribe, GPT:
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(openai.api_key, filename, "whisper-1")
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
        filename = None

    # prompt interfaces
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)

    # file section interface for prompts against large documents as context
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])


    # Document section chat
        
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice) # *************************************
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        
        #response = chat_with_model(user_prompt, ''.join(list(document_sections,)), model_choice) # *************************************

        # Divide the user_prompt into smaller sections
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            # Process each section with the model
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        
        #st.write('Response:')
        #st.write(full_response)

        response = full_response
        st.write('Response:')
        st.write(response)
        
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order

    # Added "Delete All" button
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()

    # Added "Download All" button
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
        
    # Sidebar of Files Saving History and surfacing files as context of prompts and responses
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
                
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_model(user_prompt, file_contents, model_choice)
            filename = generate_filename(file_contents, choice)
            create_file(filename, user_prompt, response, should_save)

            st.experimental_rerun()
            #st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
                
if __name__ == "__main__":
    main()

load_dotenv()
st.write(css, unsafe_allow_html=True)

st.header("Chat with documents :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    process_user_input(user_question)

with st.sidebar:
    st.subheader("Your documents")
    docs = st.file_uploader("import documents", accept_multiple_files=True)
    with st.spinner("Processing"):
        raw = pdf2txt(docs)
        if len(raw) > 0:
            length = str(len(raw))
            text_chunks = txt2chunks(raw)
            vectorstore = vector_store(text_chunks)
            st.session_state.conversation = get_chain(vectorstore)
            st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
            filename = generate_filename(raw, 'txt')
            create_file(filename, raw, '', should_save)
            #create_file(filename, raw, '')

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-large')
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'all-MiniLM-L12-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
 
    # llm = HuggingFaceHub(repo_id="stabilityai/stablelm-3b-4e1t", model_kwargs={"temperature":1, "max_length":512})
 
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    st.write(response)


###############################################################################
###############################################################################
#Below is the test code, module code can is written above octothorpes

# def extract_text_from_pdf(filename):
#     file_path = os.path.join('docs', filename)

#     try:
#         pdf_file = open(file_path, 'rb')
#     except FileNotFoundError:
#         print(f"The file '{file_path}' was not found.")
#         return

#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""

#     for page in pdf_reader.pages:
#         text += page.extract_text()

#     pdf_file.close()
#     return text

# def export_text_to_txt(pdf_filename):
#     try:
#         if not os.path.exists('texts'):
#             os.makedirs('texts')
        
#         txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'

#         txt_filepath = os.path.join('texts', txt_filename)

#         with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
#             txt_file.write(extract_text_from_pdf(pdf_filename))
#     except Exception as e:
#         print(f"An error occurred while writing to '{txt_filename}': {str(e)}")



# if __name__ == "__main__":
#     pdf_files_to_process = ['fundamental_rights.pdf', 'indian_constitution.pdf']

#     for pdf_filename in pdf_files_to_process:
#         export_text_to_txt(pdf_filename)
#         print(f"Text extracted from '{pdf_filename}' and saved as '{pdf_filename}.txt' in the 'texts' folder.")
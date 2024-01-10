import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) #creates individual pages
        #looping thorugh the pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, #contains 1000 chars
        chunk_overlap=200,
        length_function=len #helps preserving the meaning inorder to comprehend the whole sentence
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    persist_directory = "EssenceOfTraditionalKnowledgeWeek4"
    vectorstore = Chroma.from_texts(texts=text_chunks,embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    #loading the existing vectorDB
    # vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    # vectorstore.get()
    return vectorstore #this vectorstore is the DB


def get_conversation_chain(vectorstore,llm_choice):
    if(llm_choice=="OpenAI"):
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5,"max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory

    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) #this is the master variable
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if(i%2==0):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)








def main():
    
    #loading the enviroinment variables
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")

    # adding css
    st.write(css, unsafe_allow_html=True)

    # inintializing session variables

    if "conversation" not in st.session_state:
        #if app reruns, it check if the variable is already present or not
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask any question w.r.t your PDFs")
    character_count = 0
    # Event handler to update character count in real-time
    if user_question:
        character_count = len(user_question)
    # Display character count
    st.write(f"Character Count: {character_count}")

    

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        llm_choice = st.radio("Select the LLM:",["OpenAI","HuggingFace"])
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # extract pdf text

                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks

                text_chunks = get_text_chunks(raw_text)


                # create vector store
                #Two options for embedding: 1. OpenAI's Ada || 2. HuggingFace Instructor Embeddings
                #For vector DBs, we have Pinecone(cloud based - non volatile), ChromaDB, FAISS(Locally stored - volatile). 
                #Method 1
                vectorstore  = get_vectorstore(text_chunks)


                # create conversation chain
                # session state gives glonal const var effect (making variable persistant)
                st.session_state.conversation = get_conversation_chain(vectorstore,llm_choice)

        





if __name__ == '__main__':
    main()
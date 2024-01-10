import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_chat import message as st_message
from pypdf import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
# import pinecone

# SETUP BEFORE USE
load_dotenv()
oa_api_key = os.getenv('OPENAI_API_KEY')
sb_api_key = os.getenv('SUPABASE_API_KEY')
sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
supabase: Client = create_client(sb_proj_url, sb_api_key)

# for switching to pinecone
# pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="us-west1-gcp");

def generate_answer(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st_message(message.content, is_user=True)
        else:
            st_message(message.content, is_user=False)
    
    # Clears the input text
    # st.session_state["input_text"] = ""

def ingest(pdfs):
    text = get_text(pdfs)
    text_batches = get_text_batches(text)

    return text_batches

def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_batches(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 100
    )
    return text_splitter.split_text(text)

# Creates Vector DB
def init_vector_db(text_batches):
    print(text_batches)
    embeddings = OpenAIEmbeddings(openai_api_key=oa_api_key)
    vector_db = SupabaseVectorStore.from_texts(text_batches, embeddings, client=supabase, table_name="documents")
    
    # for switching to pinecone
    # vecter_db = pinecone.create_index("python-index", dimension=1536, metric="cosine")

    return vector_db

def get_conversation_chain(vector_db):
    llm = ChatOpenAI(openai_api_key=oa_api_key, model="gpt-3.5-turbo")

    # for switching to huggingface to use different LLMs
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type="similarity"),
        memory=memory
    )

    return conversation_chain


def main():
    st.header("Welcome to Jays Trainable Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.history = None
    
    user_input = st.text_input("Talk to the bot", key="input_text")
    if user_input:
        generate_answer(user_input)

    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                text_batches = ingest(pdfs)

                # Initialize the vector db
                vector_db = init_vector_db(text_batches)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_db)

if __name__ == "__main__":
    main()
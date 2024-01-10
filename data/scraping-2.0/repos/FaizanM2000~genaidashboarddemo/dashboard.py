import streamlit as st
import langchain, pinecone
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
import uuid  # <-- Don't forget to import uuid

# Streamlit Setup
st.title('Search Demo')
openai_api_key = st.secrets['OPENAI_API_KEY']
pinecone_api_key = st.secrets['PINECONE_API_KEY']

# Initialize Pinecone only once
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create a variable to hold the current index name
indexes = pinecone.list_indexes()
if indexes:  # Check that the list is not empty
    current_index_name = indexes[0]


if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to read PDF files
def read_pdf(uploaded_pdf):
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    num_pages = len(pdf_reader.pages)
    text_content = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text_content += page.extract_text()
    return text_content

# File upload logic
uploaded_file = st.file_uploader("Choose a text or PDF file", type=['txt', 'pdf'])
if uploaded_file is not None:
    new_index_name = 'docindex'  # Create a unique index name

    # Delete existing index if there is one
    pinecone.delete_index(current_index_name)

    # Update the current index name
    current_index_name = new_index_name

    # Create a new Pinecone index
    pinecone.create_index(current_index_name, dimension=1536, metric="cosine")

    
    if uploaded_file.type == "application/pdf":
        file_content = read_pdf(uploaded_file)
    else:
        file_content = uploaded_file.read().decode('utf-8')


    # Split the file content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
    )
    book_texts = text_splitter.create_documents([file_content])

    # Create Pinecone index
    with get_openai_callback() as cb:
        book_docsearch = Pinecone.from_texts([t.page_content for t in book_texts], embeddings, index_name=current_index_name)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def generate_response(input_text):
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=300)
        with get_openai_callback() as cb:
            query = input_text
            docs = book_docsearch.similarity_search(query,k=2)
            chain = load_qa_chain(llm, chain_type="stuff")
            answers = chain.run(input_documents=docs, question=query)
            
            # Append user's query and bot's answer to the session state messages
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answers})  # Make sure 'answers' is in a displayable format
    
    
    
    # Form for question input
    with st.form('my_form'):
        text = st.text_area('Enter your query:', 'Type your query here...')
        submitted = st.form_submit_button('Submit')
        if submitted:
            generate_response(text)

    # Display existing chat messages AFTER form submission
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Import necessary libraries and modules
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

# Function to read PDF content
def read_pdf(file):
    # Create a PdfReader object to read the provided PDF file
    pdf_reader = PdfReader(file)
    text = ""

    # Iterate through each page in the PDF and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Main Streamlit app
def main():
    # Set the title and create the main Streamlit interface
    st.title("Bank Statement Assistant")
    st.sidebar.title('Upload Bank statement')
    st.sidebar.markdown('''
        ## About
        Upload a Bank statement, then perform a query.
    ''')

    # Allow the user to upload a document with specific file types (PDF, DOCX, TXT)
    uploaded_file = st.sidebar.file_uploader("Choose a document to upload", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        try:
            # Read the content of the uploaded PDF document
            text = read_pdf(uploaded_file)
            
            # Display a success message and instructions for the user
            st.success("PDF successfully uploaded. The content is hidden. Type your query in the chat window.")
            
        except Exception as e:
            # Handle any errors that occur while reading the PDF
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        # Initialize a text splitter for chunking the document content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the PDF text and create a list of document chunks
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create a vectorstore
        ## -------------ADD YOUR API KEY-----------
        api_key = 'YOUR API KEY'

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        # Store the processed data in the session state
        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot using OpenAI GPT-3.5-turbo model
        llm = ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept and process user input as a query
        if prompt := st.chat_input("Ask your questions from the uploaded PDF?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Query the chatbot for a response
            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

            # Display the chatbot's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

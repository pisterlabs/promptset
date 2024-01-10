from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat




import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']




# Initialize everything outside the main function 
# to avoid re-initializing them every time the Streamlit app reruns
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
persist_directory = 'docs/chroma/'  # specify your persist directory

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """
    You are a very knowledgeable AI with access to a large amount of information about dental science.
    You are answering questions mainly to highly qualified university professors and KOLs in the dental field.
    Use the following pieces of context to answer the question at the end and include the author and year of the publication from which the context was taken in this format (autohor, year).
    If you don't know the answer, just say that you don't know, and do not attempt to fabricate an answer. 
    Keep your answers informative and comprehensive at the same time.
    Try to anticipate possible follow-up questions and incorporate those answers into your response.
    Always end your answer by saying "thanks for asking!" to maintain a friendly tone.
    {context}
    Question: {question}
    Helpful Answer:"""
)

def load_static_documents():
    static_docs = []
    for file in os.listdir('docs'):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join('docs', file))
            pages = loader.load()
            static_docs += pages
    return static_docs


# Load static documents and split
static_documents = load_static_documents() # You need to implement this function
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)
static_splits = text_splitter.split_documents(static_documents)

# Create an instance of Chroma using the from_documents class method
vectordb = Chroma.from_documents(
    documents=static_splits,
    embedding=embedding,
    persist_directory=persist_directory,
)

# Initialize retrieval and memory
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

# Streamlit app
def main():
    st.title("Chatbot App")

    # Display the vector count as soon as the page loads
    st.write("The size of the vectorstore is: ", vectordb._collection.count())

    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    pdf_directory = "docs"

    if uploaded_files:
        pages = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily
            with open(pdf_directory + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(pdf_directory + uploaded_file.name)
            pages += loader.load()
            # Remove the file after loading
            os.remove(pdf_directory + uploaded_file.name)
        
        # Split pages
        docs = text_splitter.split_documents(pages)
        splits = text_splitter.split_documents(docs)
        
        # Update Chroma instance with new documents
        vectordb.add_documents(splits)

        # Display the updated vector count
        st.write("The size of the vectorstore is: ", vectordb._collection.count())

    st.subheader("Chat with the bot")

   # Initialize the chat history in the session state
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm HugChat, How may I help you?"]

    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    def get_user_input():
        user_input = st.text_input("Type your question here...")
        return user_input
    
    def generate_bot_response(prompt):
        # Assuming you have a method to generate bot response.
        response = qa_chain({"query": prompt})
        return response["result"]

    with input_container:
        user_input = get_user_input()
        submit_button = st.button("Submit question")  # Add a submit button

    with response_container:
        if submit_button and user_input:  # Check if submit button is pressed and user input is not empty
            with st.spinner('Searching for the answer...'):  # Add a spinner
                # Generate bot response
                response = generate_bot_response(user_input)
                # Append user input and bot response to the session state
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
        
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))

    if st.button('Clear chat history'):  # Add a clear chat history button
        st.session_state.past = ['Hi!']
        st.session_state.generated = ["I'm HugChat, How may I help you?"]

# Run the app
if __name__ == '__main__':
    main()
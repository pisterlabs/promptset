import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import DuckDuckGoSearchRun
import os
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

# openai_api_key = os.environ["OPENAI_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]


avatar_path = "avatar.jpg"

st.set_page_config(
    page_title="Project Management Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Function to load and split the PDF content
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

# Paths to the PDFs
pdf_paths = ["pmbok_guide_v6.pdf", "pmbok_guide_v7.pdf"]

# Splitter and Embedding setup
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embedding_model = OpenAIEmbeddings()

# Check if vectorstore is already in session state
if "vectorstore" not in st.session_state:
    # Load, split, and store content for each PDF
    all_documents = []
    for pdf_path in pdf_paths:
        splits = load_and_split_pdf(pdf_path)
        all_documents.extend(splits)

    # Create a single vectorstore from combined documents
    vectorstore = FAISS.from_documents(documents=all_documents, embedding=embedding_model)
    st.session_state.vectorstore = vectorstore
else:
    vectorstore = st.session_state.vectorstore

# Setup the chat mechanism
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
streaming_llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                           temperature=0)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup Conversational Retrieval Chain for the combined PDF content
qa = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator,
    memory=memory)

st.title("Project Management Chatbot with Conversation Memory")

# Set up streamlit layout with containers
instructions_container = st.container()
input_container = st.container()
chat_container = st.container()

with instructions_container:
    st.header("Instructions")
    st.write("""
    - This chatbot provides answers related to the PMBOK guide.
    - Type in your question about project management in the chat input below.
    - Slide to adjust response specificity: left for broader answers, right for more focused ones.
    - For detailed queries or more context, refer to the PMBOK guide directly.
    """)

with input_container:
    # Slider to control the LLM temperature
    temperature = st.slider("Adjust chatbot specificity:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    llm.temperature = temperature

with chat_container:
    # Display chat messages from history on app rerun
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatar_path if message["role"] == "assistant" else None):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about project management or PMBOK guide:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check combined PDF content for the answer using the ConversationalRetrievalChain
        response = qa({"question": prompt, "chat_history": st.session_state.messages})
        response_text = response['answer']
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant", avatar=avatar_path):  # Add avatar for assistant messages
            st.markdown(response_text)

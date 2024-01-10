import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

st.set_page_config(page_title="Dutch Uncle AI")

st.title("Dutch Uncle AI")
st.write("Dutch Uncle is a LLM trained on the world's best financial knowledge. Ask it any financial question to receive unbiased, expert advice instantly!")

# Initialize the query history in session_state if it doesn't already exist
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Display past queries and responses
st.write("Past Queries:")
for idx, item in enumerate(st.session_state.query_history):
    st.write(f"Query {idx+1}: {item['query']}")
    st.write(f"Response {idx+1}: {item['response']}")
    st.write("---")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "What financial questions do you have?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Load in the PDF
pdf_reader = PdfReader('eypfg.pdf')

# Load the pages
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Split text into tokens that are manageable
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text=text)

# Create embeddings that are mapped from the splits
store_name, _ = os.path.splitext('eypfg.pdf')

if os.path.exists(f"{store_name}.pkl"):
    with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
else:
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

# Accept user input/question
if (prompt := st.chat_input()) is not None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Concatenate past queries and responses to use as context
    context = ""
    for item in st.session_state.query_history:
        context += f"Query: {item['query']} Response: {item['response']} "

    # Separate the current query from the context
    current_query = f"Query: {prompt} "

    docs = VectorStore.similarity_search(query=current_query, k=3)

    llm = OpenAI(temperature=0.1,)
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    # Send both context and current query to the model
    # Adjust this part according to how your LLM handles context and query
    response = chain.run(input_documents=docs, context=context, question=current_query) 

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Store the new query and its response in session_state
    st.session_state.query_history.append({
        'query': prompt,
        'response': response
    })
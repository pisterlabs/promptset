# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
'''
streamlit run streamlit_yt_QA_bot.py
'''
import os

import random
import time
import streamlit as st
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './aiap-13-ds-7e16bb946970.json'

# Initialize the YouTube loader
loader = YoutubeLoader.from_youtube_url(
    'https://www.youtube.com/watch?v=jlivBvu3Jrc',
    add_video_info=True,
    language=["en-US"])

# Load the data
data = loader.load()

# Initialize text splitter for QA
text_splitter_qa = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split text into docs for QA
docs_qa = text_splitter_qa.split_documents(data)

# Create the LLM model for the question answering
llm_question_answer = ChatVertexAI(temperature=0.2)

# Create the vector database and RetrievalQA Chain
embeddings = VertexAIEmbeddings()
db = FAISS.from_documents(docs_qa, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever())

# Streamlit application
st.title('Genshin Impact QA Bot')

# st.title("Retrieval-based QA System")

# Initialize your retrieval QA system
qa = RetrievalQA.from_chain_type(
    llm=llm_question_answer,
    chain_type="stuff",
    retriever=db.as_retriever()
)


# if "history" not in st.session_state:
#     st.session_state.history = []
#     print ("done")
#     print("st.session_state.history: ", st.session_state.history)
#     print("st.session_state.history: ", st.session_state['history'])

# # Display previous Q&A
# for q_and_a in st.session_state.history:
#     st.markdown(f"**Q:** {q_and_a['question']}")
#     st.markdown(f"**A:** {q_and_a['answer']}")

# # Get the user question
# question = st.text_input("Ask a question:")

# if question:
#     # Run the QA chain to query the data
#     answer = qa.run(question)
#     st.session_state.history.append({"question": question, "answer": answer})

#     # Display the answer
#     st.markdown(f"**Q:** {question}")
#     st.markdown(f"**A:** {answer}")

# #######################################
# if "history" not in st.session_state:
#     st.session_state.history = []

# for message in st.session_state.history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is your question?"):
#     st.session_state.history.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         # Assuming the qa.run method returns a generator of responses
#         for response in qa.run(prompt):
#             full_response += response
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.history.append(
#         {"role": "assistant", "content": full_response})
######################################################
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous Q&A
for q_and_a in st.session_state.history:
    st.markdown(f"**Q:** {q_and_a['question']}")
    st.markdown(f"**A:** {q_and_a['answer']}")

# Get the user question
question = st.text_input("Ask a question:")

if question:
    # Run the QA chain to query the data
    answer = qa.run(question)
    st.session_state.history.append({"question": question, "answer": answer})

    # Display the answer
    st.markdown(f"**Q:** {question}")
    st.markdown(f"**A:** {answer}")

# ##################################################################

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input(placeholder="Ask a question..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""

# # Render chat UI
# user_input = st.chat_input(placeholder="Ask a question...")
# if user_input:
#     chat = st.session_state.get('chat', [])

#     if len(chat) > 0 and chat[-1]['role'] == 'user':
#         question = chat[-1]['content']
#         answer = qa.run(question)
#         chat.append({'role': 'system', 'content': answer})

#     for message in chat:
#         st.chat_message(**message)

#     st.session_state['chat'] = chat

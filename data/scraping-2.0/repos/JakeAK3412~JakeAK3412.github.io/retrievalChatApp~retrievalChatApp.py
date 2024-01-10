from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import FAISS 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory 
import os
import sys
import streamlit as st

#note - export OPENAI_API_KEY as your openai key in the terminal to use this
#or do $env:OPENAI_API_KEY="yourkey" in powershell

#second note - to use easily, run this in the terminal:
#streamlit run retrievalchatapp.py pathToPDF


#check if exactly one command-line argument is provided
if len(sys.argv) != 2:
    print("Please provide exactly one command-line argument.")
    sys.exit(1)
#Here we get the path to the pdf from the command line
first_argument = sys.argv[1]

#if it isn't a path to a pdf, exit
if not first_argument.endswith(".pdf"):
    print("insert a pdf only please")
    sys.exit(1)

#Caching this so openai doesn't charge for every rerun of the app
@st.cache_data
def get_chain(pathToPDF):
   """
   This function takes in a path to a pdf and returns a 
   langchain chat chain with a vector store retriever 
   that can be used to answer questions about the pdf

   input: pathToPDF - path to the pdf to be used

   output: chain - a langchain chat chain that can be used to answer questions about the pdf
   """

   #Here we use pyPDF loader to load the text of the pdf and split by pages
   pages = PyPDFLoader(pathToPDF).load_and_split()#

   #get the embeddings from text-ada-002 and create a faiss vector store
   faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings()) 

   #get the langchain retriever object
   retriever = faiss_index.as_retriever()

      #Fun part - this is the memory object for the model,
   #it is not part of the streamlit functionality, 
   #but it will store up to a certain amount of tokens
   #and condense them when necessary
   memory = ConversationBufferMemory( 
      memory_key='chat_history', 
      return_messages=True, 
      output_key='answer')

   #create the chain
   chain = ConversationalRetrievalChain.from_llm( 
      llm=ChatOpenAI(), 
      retriever=retriever, 
      memory=memory,
      verbose=True
   )

   return chain


#Set the title of the app
st.title("Jake's overpaid assistant")

#Initialize the chat history in the session state
if "chat_history" not in st.session_state:
      st.session_state["chat_history"] = []

#Display the chat messages from the history, if we have any
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])


#Get the chain, cached so we get memory of the conversation (and I save money on openai)
chain = get_chain(first_argument)

#Get the user input - due to the nature of streamlit, this will essentially run in a loop
if user_input := st.chat_input("Ask a question"):
    
    #display new message
    with st.chat_message("Me"):
        st.markdown(user_input)
    
    #add to chat history
    st.session_state.chat_history.append({"role": "Me", "content": user_input})

    #get the answer from the chain
    answer = chain({'question': user_input})['answer'].strip()

   #display the answer
    with st.chat_message("robot"):
      st.markdown(answer)

   #add to chat history
    st.session_state.chat_history.append({"role": "robot", "content": answer})


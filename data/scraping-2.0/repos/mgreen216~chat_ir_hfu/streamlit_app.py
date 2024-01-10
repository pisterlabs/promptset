import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import tableauserverclient as TSC

st.set_page_config(page_title="Chat with the Holy Family Factbook, Blue Facts 🐯 ", page_icon="🐾", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Holy Family Factbook, Blue Facts 🐅 💬")
st.info("Check out the full Factbook at the our university page (https://public.tableau.com/app/profile/hfuieti/vizzes)", icon="📊")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Holy Family University's data!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="⏳ Loading and indexing the data 📈 – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on Holy Family University https://www.holyfamily.edu .  Assume that all questions are related to the Holy Family University and the information related to the Fact book. When a question is asked use the knowledge base knowledge_base.json to look for questions and the answers. When responding give the academic year as well as the answer. Keep your answers technical and based on facts – do not hallucinate information. Do your best to respond with a question that is similar to the one asked. Ask a clarifying question if the information is unclear. You should respond in a friendly and helpful way. When questions asked do not match was available to you, suggest a quesiton that you can answer from the knowledge base that is similar. Respond to the question with a Hmm...I don't know that yet but I'm still learning. Would you like to learn about... and insert a question from the knowledge base that you can answer. Please include tiger puns and emojis in responses to best engage people in the conversation. "))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on Holy Family University https://www.holyfamily.edu . Assume that all questions are related to the Holy Family University and the information related to the Fact book. Keep your answers technical and based on facts – do not hallucinate information.")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🧐 Thinking... "):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


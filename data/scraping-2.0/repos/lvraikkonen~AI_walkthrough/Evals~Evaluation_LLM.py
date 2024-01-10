import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru

from dotenv import load_dotenv
load_dotenv()

## Load environment variables
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACE_API_KEY"]

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

hugs = Huggingface()
openai = OpenAI() # default gpt-3.5-turbo
tru = Tru()
tru.reset_database()

# Quick Build baseline LLM chain
template = """You are a chatbot having a conversation with a human.
        {chat_history}
        Human: {human_input}
        Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(model_name="gpt-4")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

## QA relevance between overall question and answer
f_relevance = Feedback(openai.relevance).on_input_output()

## Moderation metrics on output
f_hate = Feedback(openai.moderation_hate).on_output()
f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

# TruLens Eval chain recorder
chain_recorder = TruChain(
    chain, app_id="contextual-chatbot", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
)

# Streamlit frontend
st.title("Contextual Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Record with TruLens
        with chain_recorder as recording:
            full_response = chain.run(prompt)
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

tru.run_dashboard()
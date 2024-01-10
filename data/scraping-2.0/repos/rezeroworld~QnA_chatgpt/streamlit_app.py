import streamlit as st
import openai
from prompt_engineering import answer_query_with_context
import toml
import pandas as pd

with open('config.toml', 'r') as f:
    config = toml.load(f)

df = pd.read_csv('qa_embeddings.csv', index_col=0, header=0)
q2embed = dict(zip(df.index, df.loc[:, df.columns != "answers"].to_numpy()))
q2a = dict(zip(df.index, df.loc[:,df.columns == "answers"].to_numpy()))

# App title
st.set_page_config(page_title="❔💬 Dental Aligner Q&A with ChatGPT")

# Hugging Face Credentials
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.title('❔💬 DentalGPT by Mahdi')
    st.markdown('This chat bot is powered by ChatGPT language model. It uses \
                 external knowledge to answer questions about dental aligners. \
                 In this case, prior to the Q&A, a 300 pages knowledge book was \
                 transformed into embeddings, and used to calculate a similarity \
                 metric against the presented query. Finally, the most relevent information \
                 is inserted into the prompt. The prompt is displayed with the bot answer \
                 to better understand where the answer comes from.')
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input, openai_api_key):
    openai.api_key = openai_api_key
    return answer_query_with_context(prompt_input, embeds=q2embed, answers=q2a, config=config, show_prompt=False)

# User-provided prompt
if prompt := st.chat_input(disabled=not openai_api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, openai_api_key) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
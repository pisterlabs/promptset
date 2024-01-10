# Import necessary libraries
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
import streamlit as st

# Import your chatbot model or function here
# For demonstration purposes, let's assume you have a function called `chatbot_response`

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
# repo_id ="bigscience/bloom-560m"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

template = """Question: {question}

Answer: Let's think as parson."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


def chatbot_response(message):
    # Implement your chatbot logic here
    # This is just a placeholder response
    response = llm_chain.run(message)
    wrapped_text = textwrap.fill(
        response, width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text

def main():
    st.title("Chatbot App")
    st.markdown(
        """
        <style>
        .chat-container {
            background-color: #f5f5f5;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #d3f6fd;
            padding: 10px;
            color: #000;
            margin-bottom: 5px;
            border-radius: 5px;
        }
        .bot-message {
            background-color: #e8f5e9;
            color: #000;
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 5px;
        }
        .bot-message p {
            margin: 0;
            font-weight: bold;
        }
        </style>
        """
    , unsafe_allow_html=True)

    st.markdown("Welcome to the Chatbot App!")

    # Create a sidebar for additional UI elements
    st.sidebar.title("Options")

    # Add a text input field for users to enter their messages
    user_input = st.text_input("User Input", "")

    if user_input :
        # Call the chatbot_response function and display the response
        st.markdown('<div class="chat-container"><div class="user-message">' + user_input + '</div><div class="bot-message"><p>Bot:</p>' + chatbot_response(user_input) + '</div></div>', unsafe_allow_html=True)
    

    # Add a clear button to reset the chatbot conversation
    if st.sidebar.button("Clear Chat"):
        st.empty()

if __name__ == "__main__":
    main()

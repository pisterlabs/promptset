import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

from streamlit_chat import message
from llm_helper import LLMHelper
from vector_storage import add_website_to_vector_store, add_confluence_to_vector_store, init_vector_store


# Load environment variables from .env file (Optional)
load_dotenv()

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", None)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def setup_ui():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat with Senacor')
    st.subheader('Ask anything about Senacor')


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


try:
    if "messages" not in st.session_state:  # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Ask me a question about all the websites you have added to the vector store!"}
        ]
    if 'sources' not in st.session_state:
        st.session_state['sources'] = ""
    if 'context' not in st.session_state:
        st.session_state['context'] = ""

    setup_ui()
    llm_helper = LLMHelper()
    prompt = st.text_input("Ask a question (query/prompt) about all the websites you have added to the vector store.")
    if st.button("Submit Query", type="primary"):
        response = llm_helper.standard_query(prompt)
        # question, response, st.session_state['context'], st.session_state['sources'] = llm_helper.get_semantic_answer_lang_chain(question=prompt, chat_history=[])
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    if st.session_state.messages:
        for i in range(0, len(st.session_state['messages']), 1, ):
            if st.session_state['messages'][i]['role'] == 'user':
                message(st.session_state.messages[i]["content"], is_user=True, key=str(i) + '_user',
                        avatar_style="big-smile")
            else:
                message(st.session_state.messages[i]["content"], key=str(i))


except Exception:
    st.error(traceback.format_exc())
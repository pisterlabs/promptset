from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st
from cryptography.fernet import Fernet
from langchain.chains import LLMChain
import os
from database import *
import toml,json

def init_lecture():
    st.session_state["lecture"] = st.session_state["lecture_list"][0]

def load_lecturenames():
    db = Database()
    uploaded_lectures = db.query("SELECT lecture from filestorage WHERE username = %s",
                                                    (st.session_state["username"],))
    uploaded_lectures = set([x[0] for x in uploaded_lectures])

    if not uploaded_lectures:
        st.error("**No Lecture uploaded. Go to the 'Upload Lecture' Tab on the side**")
        st.session_state["lecture_list"] = []
        st.stop()
    st.session_state["lecture_list"] = uploaded_lectures

def streamlit_setup_explainer_bot():
    return setup_explainer_bot(st.session_state["language"])

def streamlit_setup_RAG():
    return setup_RAG(st.session_state["lecture"], st.session_state["language"])

def get_default_messages():
    try:
        lecture = st.session_state['lecture']
        return [
            {
                "role": "assistant",
                "content": f"You are chatting with the {lecture} slides in {st.session_state['language']}. How can I help you?"
            }
        ]
    except KeyError:
        return [
            {
                "role": "assistant",
                "content": f"Couldn't load correctly. Do you have a valid OpenAI api token?"
            }
        ]

# This is a list instead of a dict because order of operations is important
# Values are lambdas so they are lazily evaluated
defaults = [
    # ("history", lambda: []),
    ("explainer", lambda: False),
    ("language", lambda: False),
    ("lecture_list", load_lecturenames),
    ("lecture", lambda: False),
    ("messages", lambda: []),
    #("qa", streamlit_setup_qa),
    ("chatbot", streamlit_setup_explainer_bot),
    #("authentication_status", lambda: False)
]

def initialize_session_state():
    for (key, default_value) in defaults:
        if key not in st.session_state:
            value = default_value()
            if value is not None:
                st.session_state[key] = value

#@st.cache_resource()
def setup_RAG(lecture: str, language: str):

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(f"tmp/{lecture}",embeddings=embeddings)

    template = """The following pieces of context is from a lecture slides. Use it to answer the users question. \n
    If the answer is not in the fiven context just say that you don't know, don't try to make up an answer.
    \n----------------\n{context}"""

    language_text = f"Your answer should be in {language}."

    template = template + language_text
    messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}")
    ]
    RAG_prompt = ChatPromptTemplate.from_messages(messages)



    if vectorstore is None:
        return None
    else:
        RAG = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever()
                                        ,combine_docs_chain_kwargs={"prompt": RAG_prompt}, return_source_documents=True)

        return RAG
    
#@st.cache_resource()
def setup_explainer_bot(language: str):
    try:
        llm = ChatOpenAI()
    except Exception as e:
        llm = None
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful college professor that explains difficult subjects easily understandable. "+
                "Come up with a precise answer to the question of the user"+
                ""
                f"Your answers should be in {language}" +

                """This is the previous conversation with the user: \n

                {chat_history}
                \n
                
                If the previous conversation does not help, come up with an answer without the convsersation.
                """
            
                
                ),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    if llm is None:
        conversation = None
    else:
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            #verbose=True,
        )
    return conversation

def check_secrets_file() -> bool:
    """Check if .streamlit/secrets.toml exists"""

    return os.path.isfile(".streamlit/secrets.toml")

def write_secrets_file(secrets_dict):
    secrets_dict['encryption_key'] = Fernet.generate_key().decode()
    if not os.path.isdir(".streamlit"):
        os.mkdir(".streamlit")
    if not os.path.isfile(".streamlit/secrets.toml"):
        with open(".streamlit/secrets.toml","w") as fp:
            toml.dump(secrets_dict, fp)
    st.rerun()

def render_secrets_creator():
        """Show the form to create a secrets.toml file"""

        with open("secrets_template.json", "r") as fp:
            secrets_template = json.load(fp)
        del secrets_template["encryption_key"]
        st.title("Initial Setup")
        st.write("""Either fill out all necessary secrets here or create a secrets.toml file from the
                'secrets_template.json' in the root folder. Save it in a folder called '.streamlit'
                 **More information in the README.**""")
        with st.form("secrets_writer"):
            for key in secrets_template:
                st.write(f"**{key}**")
                if key != "gmail_service_account":
                    for input in secrets_template[key]:
                        if input == "gmail_pw":
                            secrets_template[key][input] = st.text_input(input,help="**Not your normal GMAIL Password, but the generated App Password, according to the README.**")
                        else:
                            secrets_template[key][input] = st.text_input(input)
                else:   
                    st.write(""" To obtain a clients secrets json for a Google Service Account
                             follow this tutorial **until STEP 3**: https://www.labnol.org/google-api-service-account-220404 """)
                    json_file = st.file_uploader("client secrets",type="json")
            submit_secrets = st.form_submit_button()
        if submit_secrets:
            secrets_template["gmail_service_account"] = (json.load(json_file))
            write_secrets_file(secrets_template)
        st.stop()

def initialize_session_state_before_login():
    init_variables = ["authentication_status", "username"]
    for var in init_variables:
        if var not in st.session_state:
            st.session_state[var] = None
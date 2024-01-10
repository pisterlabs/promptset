import streamlit as st
from streamlit.components.v1 import html
from streamlit_extras.switch_page_button import switch_page
import openai
import os

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import TokenTextSplitter
from llama_index.indices.prompt_helper import PromptHelper
import tempfile

documents_folder = "documents"

def sidebar_stuff1():
    html_temp = """
                        <div style="background-color:{};padding:1px">

                        </div>
                        """


    button = """
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="kaushal.ai" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Support my work" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>"""
    with st.sidebar:
        st.markdown("""
        # ● About 
        "Talk to PDF" is an app that allows users to ask questions about the content of a PDF file using Natural Language Processing. 
        
        The app uses a question-answering system powered by OpenAI's GPT 🔥 to provide accurate and relevant answers to the your queries.       """)

        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown("""
        # ● Get started
        ・Paste your OpenAI API key. (click on the link to get your API key)
       
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)


        st.markdown("""
        Made by [@Obelisk_1531](https://twitter.com/Obelisk_1531)
        """)
        html(button, height=70, width=220)
        st.markdown(
            """
            <style>
                iframe[width="210"] {
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


def sidebar_stuff2():
    html_temp = """
                        <div style="background-color:{};padding:1px">

                        </div>
                        """


    button = """
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="kaushal.ai" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Support my work" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>"""
    with st.sidebar:
       
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown("""
       
        ・Choose your model (gpt-3.5-turbo or gpt-4)
        
        ・Adjust the temperature according to your needs 

        
        (It controls the randomness of the model's output. A higher temperature (e.g., 1.0) makes the output more diverse and random, while a lower temperature (e.g., 0.5) makes the output more focused and deterministic.)
       
        ・Upload a PDF file and ask questions about its content

        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)


        st.markdown("""
        Made by [@Obelisk_1531](https://twitter.com/Obelisk_1531)
        """)
        html(button, height=70, width=220)
        st.markdown(
            """
            <style>
                iframe[width="210"] {
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


def sidebar_stuff3():
    html_temp = """
                        <div style="background-color:{};padding:1px">

                        </div>
                        """


    button = """
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="kaushal.ai" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Support my work" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>"""
    with st.sidebar:
       
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown("""
      
        ・Ask questions about your documents content
        
        ・Get instant answers to your questions
       
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)


        st.markdown("""
        Made by [@Obelisk_1531](https://twitter.com/Obelisk_1531)
        """)
        html(button, height=70, width=220)
        st.markdown(
            """
            <style>
                iframe[width="210"] {
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    

def save_file(doc):
    fn = os.path.basename(doc.name)
    # check if documents_folder exists in the directory
    if not os.path.exists(documents_folder):
        # if documents_folder does not exist then making the directory
        os.makedirs(documents_folder)
    # open read and write the file into the server
    open(documents_folder + '/' + fn, 'wb').write(doc.read())
    # Check for the current filename, If new filename
    # clear the previous cached vectors and update the filename 
    # with current name     
    if st.session_state.get('file_name'):
        if st.session_state.file_name != fn:
            st.cache_resource.clear()
            st.session_state['file_name'] = fn
    else:
        st.session_state['file_name'] = fn

    return fn


def remove_file(file_path):
    # Remove the file from the Document folder once 
    # vectors are created
    if os.path.isfile(documents_folder + '/' + file_path):
        os.remove(documents_folder + '/' + file_path)


@st.cache_resource
def query_engine(pdf_file, model_name, temperature):
    with st.spinner("Uploading PDF..."):
        file_name = save_file(pdf_file)
    
    llm = OpenAI(model=model_name, temperature=temperature)

    service_context = ServiceContext.from_defaults(llm=llm)
    with st.spinner("Loading document..."):
        docs = SimpleDirectoryReader(documents_folder).load_data()
    with st.spinner("Indexing document..."):
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    with st.spinner("Creating query engine..."):
        query_engine = index.as_query_engine()

    st.session_state['index'] = index
    st.session_state['query_engine'] = query_engine
    switch_page('chat with pdf')

    return query_engine


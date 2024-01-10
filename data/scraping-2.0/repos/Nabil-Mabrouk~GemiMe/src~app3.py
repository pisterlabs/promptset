import streamlit as st
import time
from pdfminer.high_level import extract_pages
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Tru
from trulens_eval import TruChain, Feedback, Tru, LiteLLM
from langchain.chains import LLMChain
from langchain.llms import VertexAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

tru = Tru()

# Configure Streamlit page and state
st.set_page_config(page_title="GemiMe", page_icon="👨‍💼")

# Define the steps of the workflow
workflow_steps = [
    "Home",
    "Specifications",
    "Design",
    "Costing",
    "Proposal"
]

with st.sidebar:
    help='''GemiMe wil take through different steps from loading the specifications to generating a proposal. You can move from one step to another manually or let GemiMe do iy automatically'''
    st.info(help)
    with st.form("config"):
        st.header("Configuration")
        selection = st.radio("Select", ['Automatic', 'Manual'])
        gemini_api_key = st.text_input("Your Gemini API key", placeholder="sk-xxxx", type="password") 
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, format="%.1f")
        max_retries = st.slider("Max Retries", 0, 10, 2, 1)

        if st.form_submit_button("Save"):
            st.session_state.model_config = {
                "selection": selection,
                "gemini_api_key": gemini_api_key,
                "temperature": temperature,
                "max_retries": max_retries,
            }
            st.success(f"Selected model: {selection}")

def load_project_specification():

    st.session_state.file_uploaded=False
    st.write("### Step 1: Loading specification file")
    # Function to upload and display the project specification
    uploaded_file = st.file_uploader("Upload Project Specification", type=["pdf", "docx"])
    if uploaded_file is not None:
        st.write("Project Specification:")
        return uploaded_file
        #st.write(uploaded_file)
        #st.session_state.file_uploaded=True
        #for page_layout in extract_pages(uploaded_file):
        #    for element in page_layout:
        #        st.write(element)
        
def main():
    #Rendering main page
    st.title("👨‍💼 GemiMe")

    tab_home, tab_specs, tab_design, tab_cost, tab_proposal =st.tabs(["Home", "Specifications", "Design", "Costing", "Proposal"])

    with tab_home:
        intro='''A proposal engineer plays a crucial role in the process of bidding for and securing projects, 
        particularly in industries where complex technical solutions are required. '''
        st.write(intro)

    with tab_specs:
        st.write("### Step 1: Loading specification file")
        intro='''Load the specification file of the project. This file can be in pdf or docx format. You can also use one of our examples demo specifciation files below'''
        st.write(intro)
        uploaded_file = st.file_uploader("Upload Project Specification", type=["pdf", "docx"])
        # create llm
        #llm = OpenAI(temperature=0.7, model=st.session_state.model)
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GEMINI_API_KEY"])
        chain = load_summarize_chain(llm, chain_type="stuff")
        text = ""
        pdf_summary = "Give me a concise summary, use the language that the file is in. "
        pdf_title="Extract the title"
        if uploaded_file is not None:
            text = extract_text(uploaded_file)

            # Clear summary if a new file is uploaded
            if 'summary' in st.session_state and st.session_state.file_name != uploaded_file.name:
                st.session_state.summary = None
                st.session_state.title = None

            st.session_state.file_name = uploaded_file.name

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
        
            # Create embeddings
            #embeddings = OpenAIEmbeddings(disallowed_special=())
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GEMINI_API_KEY"])
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            with st.form("Chat with specifications"):
                st.header("Ask a question about the document that you uploaded")
                temp="You will be provided with a set of documents and a user question. Try to answer the user question by using the information contained in the documents. user question:"
                question=st.text_input("Enter your question:")
                question=temp+question
                submit_button = st.form_submit_button(label="Submit")

                if submit_button:
                    docs = knowledge_base.similarity_search(question)
                    llm_response=chain.run(input_documents=docs, question=question)
                    st.markdown(llm_response)



            #if 'summary' not in st.session_state or st.session_state.summary is None:
            #    try:
            #        st.session_state.summary = chain.run(input_documents=docs, question=pdf_summary)
            #        st.info(st.session_state.title)
            #        st.write(st.session_state.summary)    
            #    except Exception as maxtoken_error:
            #        # Fallback to the larger model if the context length is exceeded
            #        print(maxtoken_error)
    with tab_design:
        st.write("Hello")
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GEMINI_API_KEY"])

        full_prompt=HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Provide a helpful response with relevant background information for the following: {prompt}",
                input_variables=["prompt"],
                )
            )
        chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])
        chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)
        with st.form("Test Form"):
            st.header("Project info")
            question=st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                llm_response = chain(question)
                st.markdown(llm_response['text'])
    with tab_cost:
        st.info("Hello")


        
if __name__ == "__main__":
    main()
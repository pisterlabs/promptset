import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import openai
from PyPDF2 import PdfFileReader
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import json

# Your API Keys must be in a .env file in the root of this project
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "elevated_ambitions"

st.title("Resume Analyzer")

def analyze_resume(resume_pdf, fields_list=None, model='gpt-3.5-turbo-1106'):
    # read in the resume file, this can be done with PyPDFLoader or UnstructuredPDFLoader
    reader = PdfReader(resume_pdf)
    full_text = str()
    for page in reader.pages:
        # take the page_content and add it to the full_text
        full_text += page.extract_text()

    # Load the prompt template
    with open("prompts/resume_extraction.prompt", "r") as f:
        template = f.read()

    # Load in the extraction results template
    with open("templates/resume_template.json", "r") as f:
        resume_template = f.read()

    # Define the prompt
    prompt_template = PromptTemplate(
        template=template,
        input_variables = ['resume', 'fields_list', 'response_template']
    )
    formatted_input = prompt_template.format(
        resume = full_text,
        fields_list = fields_list,
        response_template = resume_template
    )

    # Define the LLM Chain
    chat_llm = ChatOpenAI(model)
    analysis_output = chat_llm.invoke(formatted_input)

    return json.loads(analysis_output.content)

# Create a placeholder for the buttons
button_placeholder = st.empty()

# Add two buttons to the placeholder
if button_placeholder.button('Upload a Resume'):
    uploaded_file = st.file_uploader("Upload your Resume in pdf format", type="pdf")
    # If a file is uploaded, clear the button placeholder
    if uploaded_file is not None:
        selectbox_placeholder.empty()
        file_uploader_placeholder.empty()

        # Add a button to trigger the resume analysis
        if st.button('Analyze Resume'):
            # Analyze the uploaded file
            analysis_results = analyze_resume(uploaded_file)

            # Check if the analysis results is a dictionary before trying to iterate over it
            if isinstance(analysis_results, dict):
                # For each category in the analysis results, update the corresponding field
                for category, value in analysis_results.items():
                    # If the value is a dictionary, loop through its items
                    if isinstance(value, dict):
                        for subcategory, subvalue in value.items():
                            # Replace spaces in subcategory name with underscores
                            subcategory_key = subcategory.replace(" ", "_")
                            st.text_input(subcategory, key=subcategory_key, value=subvalue)
                    else:
                        # Replace spaces in category name with underscores
                        category_key = category.replace(" ", "_")
                        st.text_input(category, key=category_key, value=value)
            else:
                st.error('The analysis results are not in the expected format. Please check the output of the analyze_resume function.')
    else:
        # Create multi-line empty boxes for each category
        categories = ["Personal Details", "Education", "Work Experience", "Projects", "Skills", "Certifications", "Publications", "Awards"]
        for category in categories:
            # Replace spaces in category name with underscores
            category_key = category.replace(" ", "_")
            st.text_area(category, key=category_key, value='', height=None)

if button_placeholder.button('Start from Scratch'):
    # Clear the file uploader and the button placeholder
    st.session_state['uploaded_file'] = None
    button_placeholder.empty()

import streamlit as st
import openai
import pandas as pd
import os  
import json
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
# from langchain.prompts import ChatPromptTemplate

import openai
from openai_function_call import OpenAISchema
import re
from pydantic import Field

class ChartDetails(OpenAISchema):
    """Cancer History"""
    mrn: str = Field(..., description="Patient's medical record number")
    last_name: str = Field(..., description="Patient's last name")
    first_name: str = Field(..., description="Patient's first name")
    age: int = Field(..., description="Patient's age")
    sex: str = Field(..., description = "Patient Sex") 
    cancer_type_1: str = Field(..., description="Type of cancer (first)")
    cancer_type_2: str = Field(..., description="Type of cancer (second)")
    cancer_type_3: str = Field(..., description="Type of cancer (third)")
    other_cancer_type_details: str = Field(..., description="Other cancer type details")
    diagnosis_dates: str = Field(..., description="Cancer diagnosis date")
    stage: str = Field(..., description="Cancer stage")
    recurrence: bool = Field(..., description="Cancer recurrence")
    recurrence_date: str = Field(..., description="Cancer recurrence date")
    recurrence_details: str = Field(..., description="Cancer recurrence details")
    alcohol_use: str = Field(..., description="Alcohol use")
    tobacco_history: str = Field(..., description="Tobacco history")
    tumor_marker_tests: str = Field(..., description="Tumor marker tests")
    treatments: str = Field(..., description="Cancer treatment")
    radiation: bool = Field(..., description="Radiation")
    radiation_details: str = Field(..., description="Radiation details")
    hormone_therapy: bool = Field(..., description="Hormone therapy")
    stem_cell_transplant: bool = Field(..., description="Stem cell transplant")
    chemotherapy: bool = Field(..., description="Chemotherapy")
    car_t_cell_therapy: bool = Field(..., description="CAR T cell therapy")
    immunotherapy: bool = Field(..., description="Immunotherapy")
    non_cancer_diagnoses: str = Field(..., description="Other medical diagnoses")
    current_medications: str = Field(..., description="Current medications")
    allergies: str = Field(..., description="Allergies")
    family_history: str = Field(..., description="Family history")
    

@st.cache_data
def parse(chart, model):
    system_prompt = """Carefully for accuracy, extract any cancer related details from medical records submitted according to the schema. Use only chart data provided. 
    For name, age, and MRN, use '***" for any that are not provided. For other fields, if data are not provided, leave blank. If extraction uncertainty exists, add 3 astersiskd (***) after the value. For example, if the patient's age is 50, but the chart is unclear, enter 50***."""
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.2,
        functions=[ChartDetails.openai_schema],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chart},
        ],
    )

    cancer_details = ChartDetails.from_response(completion)
    
    return cancer_details

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
    
st.set_page_config(page_title='Oncology Parser Assistant', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("Oncology Parser Assistant")
st.write("ALPHA version 0.2")
disclaimer = """**Disclaimer:** This is a tool to assist chart abstraction for cancer related diagnoses. \n 
2. This tool is not a real doctor. \n    
3. You will not take any medical action based on the output of this tool. \n   
"""

with st.expander('About Oncology Parser - Important Disclaimer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.info(disclaimer)
    st.write("Last updated 8/16/23")
    
if check_password():
    selected_model = st.selectbox("Pick your GPT model:", ("GPT-3.5 ($)", "GPT-3.5-turbo-16k ($$)", "GPT-4 ($$$$)"), index = 1)
    if selected_model == "GPT-3.5 ($)":
        model = "gpt-3.5-turbo"
    elif selected_model == "GPT-3.5-turbo-16k ($$)":
        model = "gpt-3.5-turbo-16k"
    elif selected_model == "GPT-4 ($$$$)":
        model = "gpt-4"
        
    st.info("📚 Let AI identify structured content from notes!" )
    st.markdown('[Sample Oncology Notes](https://www.medicaltranscriptionsamplereports.com/hepatocellular-carcinoma-discharge-summary-sample/)')
    
    col1, col2 = st.columns(2)
    with col1:
        copied_note = st.text_area("Paste your note here", height=800)
    if st.button("Extract"):
        extracted_data = parse(copied_note, model)
    # st.write(type(extracted_data))
        with col2:
            # st.text(extracted_data)
            # st.markdown(f"Extract: {extracted_data}")
            # Split the text into key-value pairs
            extracted_string = str(extracted_data)
            # Split the text into lines
            # Regular expression to match key-value pairs
            pattern = r"(\w+)=('[^']*'|[^ ]*)"

            # Find all matches in the text
            matches = re.findall(pattern, extracted_string)

            # Process each match and store in a list of tuples
            data = []
            for key, value in matches:
                # Remove the quotes around the value, if any
                value = value.strip("'")
                
                # Append the key-value pair to the data list
                data.append((key, value))

            # Convert the list of tuples to a DataFrame
            df = pd.DataFrame(data, columns=['Key', 'Value'])

            # Display the DataFrame in a table in Streamlit
            st.table(df)
import streamlit as st
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import json

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Streamlit App
st.set_page_config(
    page_title="Psychological Disorders Analyser V.2:green_heart:",
    page_icon=":green_heart:",
    layout="wide"
)

st.balloons()
st.title('Psychological Disorders Analyser V.2:green_heart:')

with st.sidebar:
    OPEN_API_KEY = st.text_input('Enter API key', type="password")
    disorder = st.text_input("Enter disorder(s) names: ")

if st.sidebar.button("Generate"):
    if not OPEN_API_KEY or not disorder:
        st.info("Please complete the missing fields to continue.")
    else:
        os.environ['OPENAI_API_KEY'] = OPEN_API_KEY
        openai.api_key = OPEN_API_KEY

        # chain 1
        chain_1_template = """Given the psychological disorder names, your job is to list all the symptoms that match with the disorder. The output \
        should be in the JSON format like below.

        Output format:
        {{
            "disorder_1" : ["symptom_1", "symptom_2", ...],
            "disorder_2" : ["symptom_1", "symptom_2", ...],
            ...
        }}

        You should list the symptoms of the disorder given only. Do not list other disorders that are not given. If there were only 1 disorder given, the dictionary \
        should have only 1 key. For example,
        {{
            "Major Depressive Disorder" : ["symptom_1", "symptom_2"]
        }}

        Steps to do it task:
        1. Extract disorder name(s) from the input. Check whether it was misspelled or not. If you do not find any disorder, please return "Please provide a valid disorder.".
        2. For each disorder, list the disorder's symptoms and add them to the result dictionary.
        3. Return only the JSON dictionary.

        The input is given below. There could be one or more disorders.
        disorder: {disorder}

        If the input string does not match any disorder, try checking whether it was misspelled or not. But if it is not similar to any disorder, just return \
        "Please provide a valid disorder.".

        Please return the output as a JSON format. Please do not return it as a python code. Run the code and give me the JSON. Give me only the JSON.
        """

        chain_1_chat = ChatOpenAI(temperature=0.0, model=llm_model)
        chain_1_prompt_template = PromptTemplate(input_variables=["disorder"], template=chain_1_template)
        chain_1 = LLMChain(llm=chain_1_chat, prompt=chain_1_prompt_template, output_key="dis_symp")

        # chain 2
        chain_2_template = '''Given the disorder names and their symptoms, your job is to generate a case, with an alias, containing all the symptoms \
        of all disorders given. Be sure not to include them straightforwardly, but include them naturally.

        The input is given below.
        {dis_symp}
        '''

        chain_2_chat = ChatOpenAI(temperature=0.5, model=llm_model)
        chain_2_prompt_template = PromptTemplate(input_variables=["dis_symp"], template=chain_2_template)
        chain_2 = LLMChain(llm=chain_2_chat, prompt=chain_2_prompt_template, output_key="case")

        # overall chain
        overall_chain = SequentialChain(
            chains=[chain_1, chain_2],
            input_variables=["disorder"],
            output_variables=["dis_symp", "case"],
            verbose=True)
        
        result = overall_chain({"disorder":disorder})

        indentation = "&nbsp;" * 8

        symp, case = st.tabs(["Symptoms", "Case"])

        if "{" not in result['dis_symp']:
            with symp:
                st.error(result['dis_symp'], icon="🚨")
            with case:
                st.error(result['dis_symp'], icon="🚨")
        elif json.loads(result['dis_symp']).get("Please provide a valid disorder.") == []:
            with symp:
                st.error("Please provide a valid disorder.", icon="🚨")
            with case:
                st.error("Please provide a valid disorder.", icon="🚨")
        else:
            with symp:
                dis_symp_dict = json.loads(result['dis_symp'])
                st.markdown("## Symptoms")
                for i, disorder in enumerate(dis_symp_dict):
                    st.markdown(f'### {i+1}. {disorder}')
                    for symp in dis_symp_dict[disorder]:
                        st.write(f'{indentation}- {symp}')

            with case:
                st.markdown("## Example Case")
                st.write(result['case'])
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import os

import random

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_csv_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.llms import OpenAI

"""
# Storytime
#### Patient Summarizer

"""

# st.markdown("# Patient Summarizer")

team_names = ['Chris Gore (Google)', 'Neha Katyal (PharmD)', 'Jonathan So (NYU)']
random.shuffle(team_names)

st.markdown("**Team (randomly shuffled each load)**:")
for name in team_names:
    st.markdown(' * ' + name)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

col1, col2 = st.columns(2)

# data_df = pd.read_csv('data.csv')

data_df = pd.read_csv('data_2pt.csv')
st.dataframe(data_df)

st.divider()

patients = list(set(data_df['PATIENT'].values))

pat_select = st.selectbox('Patient', patients, index=0,)

data_filtered_1_df = data_df[data_df['PATIENT'] == pat_select]

st.dataframe(data_filtered_1_df)

# encounters = list(set(data_df['ENCOUNTER'].values))
encounters = list(set(data_filtered_1_df['ENCOUNTER'].values))

enc_select = st.selectbox('Encounter', encounters, index=0,)

data_filtered_df = data_filtered_1_df[data_filtered_1_df['ENCOUNTER'] == enc_select]
# data_filtered_df = data_df[data_df['ENCOUNTER'] == enc_select]

st.dataframe(data_filtered_df)

if len(openai_api_key) > 0:

    st.divider()
    # Large Language Models (LLMs)
    ## Select model (note sidebar)
    model_name = st.sidebar.radio("Model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
    st.sidebar.markdown("Note. GPT-4 is recommended for better performance.")

    ## Set OpenAI API Key (get from https://platform.openai.com/account/api-keys)
    os.environ["OPENAI_API_KEY"] = openai_api_key

    ## Instantiate model
    llm = ChatOpenAI(model_name=model_name, temperature=0.5)
    
    agent = create_pandas_dataframe_agent(llm, data_filtered_df, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    summary_prompt = '''
        The following are descriptions of the dataframe's columns:

        'START' - Date the encounter started
        'STOP' - Date the encounter ended
        'ENCOUNTERCLASS' - visit type
        'DESCRIPTION’ - description of the visit type
        'REASONDESCRIPTION' - reason for the visit
        'careplansDESCRIPTION’ - description of care given
        ‘devicesDESCRIPTION - types of devices patient may have
        'imagingBODYSITE_DESCRIPTION'- description of body part for imaging procedure
        'imagingMODALITY_DESCRIPTION'- modality of imaging procedure
        'immunDESCRIPTION'- description of immunization administered in the encounter
        'medsDESCRIPTION'- description of medication administered in the encounter
        'medsREASONDESCRIPTION'- reason for medication administration
        'obsDESCRIPTION'- description of medical observation
        'obsVALUE’- numerical value for medical observation
        'obsUNITS'- unit for the numerical value of each medical observation
        'procDESCRIPTION'- description of procedure patient underwent
        'procREASONDESCRIPTION'- reason for procedure patient underwent

        Summarize using the following format:

        Summary: Summarize this patient’s encounter as a clinician.

        Simple summary:  Summarize this encounter in 2-3 sentences as a layman.
    '''


    if st.button("Sumarize!", key="prompt_chain_button"):
        ## Spinner
        with st.spinner("Running"):
            ## Create prompt based on template
            # prompt = PromptTemplate(
            #     input_variables=["foo", "bar"],
            #     template=template,
            # )

            ## Load LLM and prompt to chain
            # chain = LLMChain(llm=llm, prompt=prompt)

            ## Run chain
            output = agent.run(summary_prompt)
            # output = chain.run({"score": score, "sex": sex})

            st.info(output)

st.divider()
with st.expander("Disclaimer"):
    st.markdown("""Please read the following disclaimer carefully before using this medical application (the "App").

This App is intended for informational and educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. The information, content, and tools provided in this App are not intended to replace or modify any information, services, or treatment provided by a qualified healthcare professional.

The developers and creators of this App make no representation, warranty, or guarantee, either expressed or implied, regarding the accuracy, completeness, or appropriateness of the information, content, or tools found within this App. Furthermore, we expressly disclaim any liability, loss, or risk incurred as a direct or indirect consequence of the use, application, or interpretation of any information provided in the App.

You, the user, are solely responsible for determining the value and appropriateness of any information or material available through this App. It is crucial to always seek the advice of a physician, medical professional, or other qualified healthcare providers with any questions, concerns or symptoms you may have regarding your health or any medical condition. Never disregard, avoid, or delay seeking appropriate medical attention because of something you have read or learned through this App.

If you believe you have a medical emergency, call your healthcare provider or emergency services immediately. This App should not be relied upon in urgent or emergency situations. It is essential to rely on the advice of qualified healthcare professionals to assess and address your specific health needs.

By using this App, you hereby agree to indemnify and hold the developers, creators, and any affiliated parties harmless from any liability, loss, claim, or expense (including reasonable attorney's fees) arising out of or related to your use of this App or its contents.

This App may contain links to third-party websites or services. We do not control, endorse, or assume any responsibility for the content, privacy policy, or practices of such websites or services. You acknowledge and agree that this App's developers and creators shall not be responsible or liable, directly or indirectly, for any damage or loss caused by, or in connection with, the use of or reliance on any site or service.

The developers and creators of this App reserve the right to modify or discontinue the App, or any features therein, at any time, without notice.

By using this App, you agree to be bound by this Disclaimer. If you do not agree with any part of this Disclaimer, please refrain from using the App.
    
    """)


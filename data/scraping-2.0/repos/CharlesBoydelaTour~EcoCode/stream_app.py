import os
import json
import requests
import openai
import numpy as np
import streamlit as st
import pandas as pd

from datetime import datetime
from transformers import GPT2TokenizerFast
#from streamlit_webrtc import WebRtcMode

from src.utils_app import get_sim_data, request_data
import src.pipeline_process as pp
from io import StringIO

st.set_page_config(layout="wide")

# OpenAI creds:
openai.organization = "org-0hLSeCcoBZl9cuiCloL2uPRC"
openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETIONS_MODEL = "text-davinci-002"
headers = {
    "X-BLOBR-KEY": "JBWNZtcWJDPl0GmgCtTLDgQJoHb6rNeE"
}

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""

aws_data = pd.read_csv('impact.csv')
aws_data = aws_data[aws_data['provider']=='aws']
aws_data = aws_data[['region', 'country_name', 'city', 'impact']]

REGION_PROMPT = """

Consider the table below:
{pd_table}
-----
1. In the table above, what are the three closest cities to {region}?
2. Sort these by lowest impact? Be sure to include the impact associated.
3. What is your recommendation for the best climate friendly option? lowest impact

Answers:

"""

def question_page():
    st.header("Q&A")
    st.write("**Ask a question about the climate friendliness of the diferent hosting regions**")
    temp = st.number_input('Temperature of answer')
    n_char = st.number_input('Length of answer', 100)
    form2 = st.form("comment2")
    question = form2.text_area("Question")
    #comment = form.text_area("Comment")
    submit2 = form2.form_submit_button("Get an answer")

    if submit2:
        st.write("Answer:")
        text = get_gpt_answer(question, temp, int(n_char))
        st.write(text)

def main_page():
    st.header("Region Summarizer")
    st.dataframe(aws_data)
    st.write("**Input your region for best climate friendly options**")
    form = st.form("comment")
    region = form.text_input("Region")
    #comment = form.text_area("Comment")
    submit = form.form_submit_button("Submit")

    if submit:
        st.write("Submitted")
        formated_prompt = REGION_PROMPT.format(
                                pd_table=aws_data.to_string(), region=region)
        text = get_gpt_answer(formated_prompt)
        st.write(text)

def request_page(sim=False):
    st.header("GPT-3 + Electricity Maps")
    st.write("Carbon Intensity Forecast for next 24hrs")
    #zone = st.text_input("zone")
    url = "https://api-access.electricitymaps.com/tw0j3yl62nfpdjv4/carbon-intensity/forecast?zone="
    live_data = st.checkbox('Use Live Data')
    if not live_data:
        forecast_df = get_sim_data()
    else:
        forecast_df = request_data(url, headers)
    str_data = forecast_df[['carbonIntensity', 'hour', 'date', 'price','Dollar per carbon']].to_string(index=False)
    st.write("**Forecast per hour of region**")
    st.dataframe(forecast_df)
    df_plot = forecast_df[['datetime','carbonIntensity','price', 'Dollar per carbon']].set_index('datetime').copy()
    # Plot the figure
    #st.pyplot(fig)
    st.line_chart(df_plot)
    FORECAST_PROMPT = """
Answer the following as truthfully as possible. 
If you are un

Consider the following forecast:
{data_string}
---
1.Summarize the data
2.Recommend a time with lowest price and carbon to run a script for an hour.


    """.format(data_string=str_data)
    temp = st.number_input('Temperature of answer')
    n_char = st.number_input('Length of answer', 100)
    if st.button('Get recommendation'):
        print(FORECAST_PROMPT)
        #st.write("temp {}, n_char {}".format(temp, n_char))
        text = get_gpt_answer(FORECAST_PROMPT, temp, int(n_char))
        print(text)
        st.write(text)


def get_gpt_answer(prompt, temperature=0.0, max_tokens=300):
    return openai.Completion.create(
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    model=COMPLETIONS_MODEL)["choices"][0]["text"].strip(" \n")

def suggest_based_on_current():
    st.header("Suggestion of Timing for Current Jobs ")
    live_url = "https://api-access.electricitymaps.com/tw0j3yl62nfpdjv4/carbon-intensity/latest?zone=DK-DK2"
    #response = requests.get(url, headers=headers)
    #latest_data = json.loads(response.text)
    carbonIntensity = 344#latest_data['carbonIntensity']
    st.write("Current carbon Intensity: {}".format(carbonIntensity))

    str_datetime = '2022-11-13T16:00:00.000Z'#str(latest_data['datetime'])

    forecast_url = "https://api-access.electricitymaps.com/tw0j3yl62nfpdjv4/carbon-intensity/forecast?zone=DK-DK2"
    forecast_df = request_data(forecast_url, headers)
    forecast_df['pct_diff'] = carbonIntensity/forecast_df['carbonIntensity']
    
    str_data = forecast_df[['carbonIntensity', 'hour', 'date', 'price', 'Dollar per carbon']][:24].to_string(index=False)
    str1 = """The forecast for the next hours is in the following table:
{fore_data}
------
The carbon intensity given in gCO2eq/kWh.


At {time}, We have been running a python script for half hour
with a carbon intensity of {carbonIntensity}.
1. Summarize the data.
2. Based on price and pct_diff suggest the best time to run the same job.
Reference the relevant data for the suggestion. Remmember we want low carbon emissions.
3. How much emissions would we have reduced if we ran it then?

Lets take this step by step.
""".format(time=str_datetime, carbonIntensity=carbonIntensity, fore_data=str_data)
    st.title("Current Status")
    st.write("""At {time}, We have been running a python script for half hour with a carbon intensity of {carbonIntensity}.
1. What would be the best time to run our script again with the lowest carbonIntesity? Reference the carbonIntensity.
2. How much emissions would we have reduced if we ran it then? Use the pct_diff column
Recommendation:
            """.format(time=str_datetime,carbonIntensity=carbonIntensity))
    temp = st.number_input('Temperature of answer')
    n_char = st.number_input('Length of answer', 100)
    if st.button('Get recommendation'):
        print(str1)
        text = get_gpt_answer(str1, temp, int(n_char))
        print(text)
        st.write(text)          

def code_reco():
    st.header("Code Recommendation")
    col1, col2 = st.columns(2)
    #put title at column 1 (Input) and column 2 (Output)
    with col1:
        st.title('Input')
    with col2:
        st.title('Recommendation')
    #set a separator element between the two columns
    st.markdown('---')

    ## Create 2 boxes in the page
    #1st box is input code, 2nd box is output code
    #input code either text or file
    col1.subheader('Add country code and Input Code (Text or File)')
    country_code = col1.text_input("Enter your country code")

    code = col1.text_area("Input code", height=500)
    code2 = col1.file_uploader("Upload file", type=['py'])

    #output code
    code_corrected = col2.code('''#Output Code''', language="python")

    #When the button is clicked, the code is processed and the output is displayed in the 2nd box
    if code:
        corrected_code_str, emissions_notcorrected, emissions_corrected, percent_reduction = pp.process(code_to_correct=code, country=country_code)
        code_corrected.code(corrected_code_str, language="python")
        col2.text("Co2 emission saved: {}%".format(percent_reduction))
        col2.text("Co2 emission left with code based in {}: {} kg".format(country_code, emissions_notcorrected))
    if code2:
        stringio = StringIO(code2.getvalue().decode("utf-8"))
        code_to_read = stringio.read()
        corrected_code_str, emissions_notcorrected, emissions_corrected, percent_reduction = pp.process(code_to_correct=code_to_read, country=country_code)
        code_corrected.code(corrected_code_str, language="python")
        col2.text("Co2 emission saved: {}%".format(percent_reduction))
        col2.text("Co2 emission left with code based in {}: {} kg".format(country_code, emissions_notcorrected))

def main():
    
    page_names_to_funcs = {
    "Main Page": main_page,
    "Ask a question": question_page,
    "GPT-3 + Electricity Maps API ": request_page,
    "Current and Forecast": suggest_based_on_current,
    "Code Recommendation": code_reco
    }
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

main()

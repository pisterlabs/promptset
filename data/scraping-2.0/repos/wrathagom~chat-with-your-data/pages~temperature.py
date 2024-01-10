import openai
import streamlit as st
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

st.title('Temperature')

st.write('If you did not notice already, the chat app queries Elastic, but it does so with SQL queries. That is because SQL is easier to generate than Elastic DSL. While I was playing around with response formatting I created this simple app to play with temperature.')

st.divider()


with st.form("Temperature Test"):
    systemInput = st.text_area("System Prompt", "Hello, world!")
    userInput = st.text_area("User Prompt")
    col1, col2, col3 = st.columns(3)

    with col1:
        minTemp = st.number_input('Min Temp', value=0.0, min_value=0.0, max_value=2.0)
    with col2:
        step = st.number_input('Temp Step', value=0.1, min_value=0.0, max_value=1.0)
    with col3:
        maxTemp = st.number_input('Max Temp', value=2.0, min_value=0.0, max_value=2.0)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Go")

if submitted:
    for x in np.arange(minTemp, maxTemp, step):
        messages = []

        if systemInput != None and systemInput != "":
            messages.append({"role": "system", "content": systemInput})

        messages.append({"role": "user", "content": userInput})


        full_response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=False,
            temperature=x
        )

        st.write('Temperature {} - {}'.format(x, full_response['choices'][0]['message']['content']))
        # st.write("slider", slider_val, "checkbox", checkbox_val)
import openai as ai
import os as os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

ai.api_key = os.getenv('OPENAI_API_KEY')

st.sidebar.title('Configurations')
st.header('This is an Explainer App')
st.subheader('This app uses OpenAI API to summarize text entered into the text box into a summary')
st.sidebar.text('Creativity')
creativity = st.sidebar.slider('How creative do you want your summary to be?', 0.0, 1.0, 0.3)
number_of_completions = st.sidebar.slider('How many summaries do you want?', 1, 10, 1)
length_of_response = st.sidebar.slider('How long do you want your summary to be?', 1, 300, 60)
frequence_penalty = st.sidebar.slider('How frequent do you want your summary to be?', 0.0, 1.0, 0.0)
presence_penalty = st.sidebar.slider('How present do you want your summary to be?', 0.0, 1.0, 0.0)
text_input = st.text_area('Enter your text here',height=200)
clicked_button  = st.button('click me') 
print(text_input)

if clicked_button:
    response = ai.Completion.create(
        engine="davinci", 
        top_p=creativity,
        prompt=text_input, 
        n=number_of_completions,
        frequency_penalty=frequence_penalty,
        presence_penalty=presence_penalty,
        max_tokens=length_of_response)
    st.write(response.choices[0].text)
    st.write(response)


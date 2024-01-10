import streamlit as st
import openai
import os
from streamlit_extras.stoggle import stoggle


openai_api_key = st.secrets["openai_key"]

# Set the OpenAI API key
openai.api_key = openai_api_key



def generate_response(prompt, user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    )

    return response.choices[0].message['content']

def app():
    st.title("The spoookiest time of the year!")
    st.subheader("👻🎃💀")

    st.markdown("## History of Halloween")
    # Add a checkbox to show or hide the history section
    stoggle(
        "Show the History of Halloween!",
        """'<iframe src="https://blogs.loc.gov/headlinesandheroes/2021/10/the-origins-of-halloween-traditions/" width="100%" height="450"></iframe>', unsafe_allow_html=True)"""
    
    )
    st.markdown("---")  # Add horizontal line for separation
    st.markdown("## Halloween Recipe and Decoration Generator")
    # Dropdown menu for user to select recipe or decoration
    option = st.selectbox('What would you like help with?', ['Halloween Recipe', 'DIY Halloween Decoration'])
    user_prompt = st.text_input('Enter more specifics (optional)')

    if st.button('Get Ideas'):
        if option == 'Halloween Recipe':
            system_prompt = "You are an expert chef who specializes in creating fun and spooky Halloween recipes."
            with st.spinner("Cooking up the recipe..."):
                response = generate_response(system_prompt, user_prompt)
                st.write(response)
        else:
            system_prompt = "You are a skilled craftsman with a knack for DIY Halloween decorations."
            with st.spinner("Searching the craft box..."):
                response = generate_response(system_prompt, user_prompt)
                st.write(response)

app()

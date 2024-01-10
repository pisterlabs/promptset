import openai
import streamlit as st
import pandas as pd



# Set up the sidebar
st.sidebar.header('API Key') 
api_key = st.sidebar.text_input('Enter your API key')
#user_api_key = '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'
st.sidebar.markdown('[Get an OpenAI API key](https://platform.openai.com/account/api-keys)')
st.sidebar.markdown('[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)')

 

# Set up the main content
st.header('Terminology to Thai') 

# Add a description
st.markdown('''
You can paste a paragraph in English and this app will translate the terminology to Thai. Also 
with the vocabulary explanation in English.
 ''')


# Create a text box for the client to submit the paragraph
paragraph = st.text_area('Enter the paragraph')

#prompt=
# Prompt is: 
#Act like an vocabulary specialist who can do the following things:
#1.read the paragraphs which contains specific vocabulary such as medical vocabulary, scientific vocabulary or else
#. The paragraph will be submitted by a client, so the python code must have a text box and a submit button. 
#Also the webpage must have sidebar which has an area to put APIkey in it.
#2.after receiving a paragraph by click submit button, the program should summarize the text and captures 
#the list of specific vocabulary and then create a pandas.dataframe to show the following information: 
#1.easy to understand meaning and 2.that word translated into Thai.

# Add a submit button
if st.button('Submit'):
    # Process the paragraph using the API key
    if paragraph:
        # Summarize the text
        openai.api_key = api_key
        #openai.api_key = user_api_key
        response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": paragraph},
    ],
)
        summary = response.choices[0].text.strip()

        # Extract specific vocabulary 
        terminology_list = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8"]
        half_length = len(terminology_list) // 2
        vocabulary = terminology_list[:half_length]

        # Create a DataFrame to show the information
        data = {
            'Vocabulary': vocabulary,
            'Meaning (English)': ['Meaning 1', 'Meaning 2', 'Meaning 3'],
            'Meaning (Thai)': ['Translation 1', 'Translation 2', 'Translation 3']
        }
        df = pd.DataFrame(data)

        # Display the DataFrame
        st.dataframe(df)
        st.warning('Please enter a paragraph before submitting.')
    pass

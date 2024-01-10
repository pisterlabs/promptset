import openai
import pandas as pd
import json
import streamlit as st


# Sidebar for OpenAI API key input
api_key = st.sidebar.text_input('Enter your OpenAI API Key', type='password')

openai.api_key = api_key

prompt = """You will receive the lyrics and you should give 10 interesting vocabularies from this lyrics .
            List the vocabulary in a JSON array, one vocab per line.
            Each vocaburaly should have 5 fields:
            - "word" - the word itself
            - "parts of speech" - the part of speech of the word
            - "translation" - the translation of the word in thai
            - "meaning" - the meaning of the word
            - "example sentence" - an example of using the word in the lyrics
            Don't say anything at first. Wait for the user to say something."""   
    
# Streamlit app
st.title("Interesting Vocaburaly from Song Lyrics")

# User input for song name and artist
lyrics = st.text_area("Enter the song name:")

# Display lyrics and interesting vocabulary on button click
if st.button("Get Vocabulary"):
    if lyrics:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {'role': 'user', 'content': lyrics}],
            max_tokens=800
        )

        st.markdown('**Vocab**')

        vocabulary_dictionary = response.choices[0].message.content
        vocabulary_dictionary_2 = json.loads(vocabulary_dictionary)
        vocabulary_df = pd.DataFrame(vocabulary_dictionary_2)
        vocabulary_df.index = range(1, len(vocabulary_df) + 1)

        st.table(vocabulary_df)

    else:
        st.warning("Please enter both the song name and artist.")
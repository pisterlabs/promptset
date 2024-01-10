
import streamlit as st
import openai
import json
import pandas as pd


# Get the API key from the sidebar called OpenAI API key
#user_api_key = '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'
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """From the passage given, extract some words, limited to 30 words.
            Translate them into Spanish,
            and indicate whether they are a verb noun adjective or adverb, 
            and order them from easiest to hardest.
            List the suggestions in a JSON array, one suggestion per line.
            Each suggestion should have 3 fields:
            - "word" - the word in English
            - "translation" - the translation from English to Spanish
            - "part of speech" - one of "verb", "noun", "adjective", "adverb"
            Don't say anything at first. Wait for the user to say something.

        """


st.title('My good words')
st.markdown('Input the the passage. \n\
            The AI will give you word, translation and part of speech.')

user_input = st.text_area("Enter some text to correct:", "Your text here")



# submit button after text input
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the response from the AI in a box
    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content


    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)


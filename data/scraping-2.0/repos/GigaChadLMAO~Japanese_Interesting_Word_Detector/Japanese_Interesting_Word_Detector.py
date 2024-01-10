import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
  
prompt = """Act as an AI Japanese text's Japanese word detector. You will receive a 
            piece of writing and you should give Japanese word detail.
            List the word detail in a JSON array with 4 columns.
            Each word detail should have 4 fields:
            - "word" - japanese word
            - "romaji" - the word's romaji
            - "word level" - word level (EASY, MEDIUM, HARD)
            - "translation" - word's translation
            Don't say anything at first. Wait for the user to say something.
        """    



st.title('Japanese Interesting Word Detector')
st.markdown("""Input Japanese text that you want to Search for interesting word. \n\
            The AI will give you the word's romaji, word level and its translation.""")

user_input = st.text_area("Enter Japanese text to search:", "Your text here")


# submit button after text input
if st.button('Search'):
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

import streamlit as st
import openai
import json
import pandas as pd

#get user the api key from the sidebar
openai_api_key = st.sidebar.text_input("Enter your API key", type="password")

client = openai.Client(api_key= openai_api_key)
prompt = """Act as an AI writing tutor in English. You will receive 
a piece of article from a user and your job is to bring the interesting word out of the article and give word's part of speech and synonym.
     list of suggestions in a JSON array, one suggestion per line. 
     Each suggestion have 3 fields:
     -"text input" - the text that should be replaced
     -"word" - word that interestingly 
     -"part of speech" - the part of speech of the word
     -"word synonym" - the synonym of the word in any part of speech 
     Don't say anything at first. Wait for the user to say something.
 """

st.title("English tutor")
st.header("Welcome to your English tutor")
st.markdown("Input the writing that you want to know more interesting words.\n\
            The AI will give you word that interestingly and its part of speech and synonym.\n\
            word that interestingly such as creepy scary.")
user_input = st.text_area("Enter your writing here", "Your text here", height=300)

# submit button after text input
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = messages_so_far
    )
    # Show the response from the AI in a box
    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content


    sd = json.loads(suggestion_dictionary)

    print (suggestion_dictionary)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)
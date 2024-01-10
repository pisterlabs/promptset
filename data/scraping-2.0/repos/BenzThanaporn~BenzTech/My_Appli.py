import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """
    Act as an AI tutor in English.
    You will receive an English passage and you should make 7 questions from the passage.
    List the questions in a JSON array, one question per line.
    Each question should have 5 fields :
    - “Question” - the question you make from the passage
    - “Answer” - the answer for the question
    - “Interesting word” - give 1 interesting word from the question
    - “Part of speech” - the interesting word's part of speech
    - “Translation in Thai” - the interesting word's translation in Thai
    The interesting word must be unique in each question.
    """
    
st.title("Let's learn from English passage")
st.markdown("Input your English passage.\n\
        The AI will make 7 questions from the passage you gave, along with their answers.\n\
        In each question, 1 interesting word and its part of speech and its translation in Thai will also be provided.\n\
        The interesting word is the word that you should learn from the question.")

user_input = st.text_area("Put English passage here")

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
    result_dictionary = response.choices[0].message.content


    rd = json.loads(result_dictionary)

    print (rd)
    result_df = pd.DataFrame.from_dict(rd)
    print(result_df)
    st.table(result_df)
    







import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as a Korean-English-Thai translator. You will recieve a Korean-english song lyrics.
            Then perform the following actions:
            1-Translate the korean-english lyrics into Thai.
            2-Write a story about the main idea of the lyrics in Thai.
            3-Search for interesting words in the lyrics that useful for TOPIK 2 Korean proficiency test and part of speech of the word.
            4-Give Korean example use case of the interesting words.
            5-Output an interesting words json object that contains the following keys: interesting words, tranlation of the word in Thai, part of speech of the word, use case of the word.
            Your answer should be a JSON object containing translation, story, output an interesting words json.
            Use the following format:
            {"Translation": "<Thai translation of the lyrics>",
            "Story": "<story about lyrics>",
            "Output interesting words JSON": <interesting word, translation of the word in Thai, part of speech, korean use case>}"""


st.title('Learning Korean with your favorite KPOP song!')
st.markdown('Input the lyrics of your favorite singer. \n\
            AI will help you understand more about the song.<3')

user_input = st.text_area("Put your lyrics:", "Your lyrics here")


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


    sd = json.loads(suggestion_dictionary["Output interesting words JSON"])

    print('คำแปล: ' + sd["Translation"])
    print("\nเพลงนี้เกี่ยวกับอะไร?: " + sd["Story"])
    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)
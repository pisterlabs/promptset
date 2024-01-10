
# This is PA5_LLM_application coding space

import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as an expert translator in Chinese-English. You will receive a 
            writing in Chinese, then you should do these following instructions.
            1. Read the writing carefully.
            2. Finding the words which is in HSK4-6 levels in the receivied writing. Listed them out. Then, Find each word's pinyin, part of speech and give its meaning in English. Label the word with its HSK level, one of "HSK4", "HSK5", "HSK6" too. 
            The writing is delimited by <TEXT> and </TEXT> tags.

            List the them in a JSON array.
            The suggestion should be in the form which is one suggestion per line.
            Each line should have these fields:
            - "Words" - the word which is in HSK4-6 levels
            - "Pinyin" - the pinyin of the word
            - "Parts of speech" - part of speech of the word, one of "名词", "动词", "形容词", "副词", "介词", "代词", "连词", "助词", "叹词", "数词", "量词", "拟声词", "缩略词", "前缀", "后缀", "语气词", "习语", "成语", "惯用语", "俗语", "谚语", "歇后语", "词组", "其他"
            - "Meaning" - the meaning of the word in English
            - "HSK" - the HSK level of the word, one of "HSK4", "HSK5", "HSK6"
            Make sure that the suggestions should be sorted by the HSK level of the word, from "HSK4" to "HSK5" to "HSK6".

            Don't say anything at first. Wait for the user to say something.
        """   

# This is the main function in this application
st.title("PA5 LLM application")
st.markdown("""This is an application which you can find bunch of interesting chinese words from the chinese writing.\n
            The application will give you word - pinyin - parts of speech - meaning - HSK level.""")
st.markdown("Please enter your writing in Chinese below.")

# Get the user input
user_input = st.text_area("Your writing in Chinese", height=300)

# Submit button after user input
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )

    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content


    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)





















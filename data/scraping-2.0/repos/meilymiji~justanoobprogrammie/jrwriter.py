import streamlit as st
import openai
import json
import pandas as pd

user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)

prompt1 = """Act as a high school student writing a paragraph in English. You will recieve a topic that contains the topic's objective and some background information. Write a paragraph that related to the topic. The paragraph should be at least 80 words. You can write more than one paragraph. The words should be at least 8 words."""

st.title('Your Junior Writer')
st.markdown('Input the topic. \n Do not forget to put an objective and some background information in the topic :)')

user_input = st.text_area("Enter an objective and some background information")
#ไม่อยู่ในกล่องข้อความ
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt1},
        {'role': 'user', 'content': user_input},
    ]
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )

    paragraph = response1.choices[0].message.content

    prompt2 = """Act as a high school teacher. You will receive a paragraph. List the words that a high school student should know from the given paragraph in JSON array. Each word should have the part of speech of that word ,the difficulty of that word,  the meaning of that word and the example usage in other context. The list should be at least 8 words."""

    messages_so_far = [
        {"role": "system", "content": prompt2},
        {'role': 'user', 'content': paragraph},
    ]
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )

    st.write('**Paragraph:**')
    st.write(paragraph)

    st.markdown('**Vocabulary List:**')
    vocab_dict = response2.choices[0].message.content
    sd = json.loads(vocab_dict)

    vocab_df = pd.DataFrame.from_dict(sd)
    st.write(vocab_df)

    st.write("**Please do not copy all of the paragraph, try to write it by yourself first. You can do it:)**")
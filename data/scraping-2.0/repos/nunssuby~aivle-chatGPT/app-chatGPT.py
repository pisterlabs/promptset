import os
import openai
from PIL import Image
import streamlit as st
openai.api_key = "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"

st.set_page_config(
    page_title="ChatGPT",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def openai_completion(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      max_tokens=150,
      temperature=0.5
    )
    return response['choices'][0]['text']

main_image = Image.open('static/main_banner.png')

st.image(main_image,use_column_width='auto')
st.title("📄 ChatGPT 🏜 Streamlit")


input_text = st.text_area("Please enter text here... 🙋",height=50)
chat_button = st.button("Do the Magic! ✨")

if chat_button and input_text.strip() != "":
    with st.spinner("Loading...💫"):
            openai_answer = openai_completion(input_text)
            st.success(openai_answer)
else:
    st.warning("Please enter something! ⚠")
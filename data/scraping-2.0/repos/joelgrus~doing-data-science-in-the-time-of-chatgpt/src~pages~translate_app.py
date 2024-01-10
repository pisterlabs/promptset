import streamlit as st
import openai
import os

# Replace with your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]


def translate_text(text, target_language):
    translation_prompt = (
        f"Translate the following English text to {target_language}:\n\n{text}"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=translation_prompt,
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].text.strip()


# Streamlit UI
st.title("Language Learning Companion")

language_choice = st.selectbox(
    "Choose the language you want to learn:",
    ["Spanish", "French", "German", "Japanese", "Klingon", "Bird speak"],
)

user_input = st.text_area("Type your message in English:")

if st.button("Translate"):
    if user_input:
        with st.spinner("Translating..."):
            translated_text = translate_text(user_input, language_choice)
        st.subheader("Translated Text:")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")

# To run the Streamlit app, save this code in a file named language_app.py and run `streamlit run language_app.py`

import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function to translate text using GPT-3.5-turbo model
def translate_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following English text to German:\n\n{text}\n\nTranslation:",
        max_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        n=1,
    )
    translation = response.choices[0].text.strip() # type: ignore

    return translation

# Streamlit app
def main():
    st.title("English to German Translation")
    st.write("Enter the English text you want to translate:")
    input_text = st.text_area("Input", height=200)
    if st.button("Translate"):
        if input_text:
            translation = translate_text(input_text)
            st.write("German Translation:")
            st.write(translation)
        else:
            st.write("Please enter some text to translate.")

if __name__ == "__main__":
    main()

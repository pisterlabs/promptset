import streamlit as st
import openai

# Set up OpenAI API credentials
openai.api_key = "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"

# Function to translate text using GPT-3.5 Turbo model
def translate_text(text, target_language):
    if target_language == "German":
        prompt = f"Translate the following English text to German:\n\n{text}"
    elif target_language == "French":
        prompt = f"Translate the following English text to French:\n\n{text}"
    elif target_language == "Spanish":
        prompt = f"Translate the following English text to Spanish:\n\n{text}"
    else:
        return "Invalid language selection"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

# Streamlit app code
def main():
    st.title("English Translation")
    input_text = st.text_input("Enter text in English")
    target_language = st.radio("Select target language", ("German", "French", "Spanish"))
    
    if st.button("Translate"):
        if input_text:
            translation = translate_text(input_text, target_language)
            st.write("Translation:")
            st.write(translation)
        else:
            st.write("Please enter some text to translate")

if __name__ == "__main__":
    main()

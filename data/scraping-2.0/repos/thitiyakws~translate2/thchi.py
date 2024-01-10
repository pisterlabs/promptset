import streamlit as st
import pandas as pd
import openai

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Sidebar to input OpenAI API key
api_key = st.sidebar.text_input("Enter OpenAI API Key")

# Function to translate Thai to Chinese and extract interesting words
def translate_and_extract_words(thai_text):
    # Define the prompt for translation
    prompt = f"Translate the following Thai sentence to Chinese: '{thai_text}'"

    # Call OpenAI ChatGPT API for translation
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )

    # Extract the translated text from the response
    chinese_translation = response.choices[0].text.strip()

    # Extract interesting words using your own logic
    interesting_words = ["word1", "word2", "word3"]  # Replace with your logic

    return chinese_translation, interesting_words

# Main content
st.title("Thai to Chinese Translator with Vocabulary Extractor")

# Input for user
thai_sentence = st.text_area("Enter your Thai sentence:")

# Translate and display results
if st.button("Translate and Extract Vocabulary"):
    if api_key:
        openai.api_key = api_key
        chinese_translation, interesting_words = translate_and_extract_words(thai_sentence)

        st.subheader("Chinese Translation:")
        st.write(chinese_translation)

        # Create DataFrame for interesting words
        df_interesting_words = pd.DataFrame({
            "Word": interesting_words,
            "Translation": [""] * len(interesting_words),
            "Example": ["Example 1", "Example 2", "Example 3"] * (len(interesting_words) // 3)
        })

        st.subheader("Interesting Words in Chinese:")
        st.dataframe(df_interesting_words)

        # Download interesting words as CSV
        csv_file_interesting_words = df_interesting_words.to_csv(index=False)
        st.download_button(
            label="Download Interesting Words as CSV",
            data=csv_file_interesting_words,
            file_name="interesting_words.csv",
            key="download_button_interesting_words"
        )
    else:
        st.warning("Please enter your OpenAI API Key in the sidebar.")

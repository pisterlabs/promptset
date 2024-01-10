
#Auther Ajit Dash dt: May 20th 2023 
#MAKE SURE TO INSTALL "pip install black" , "pip install streamlit-black-theme" and "pip install streamlit" in the terminal
# Authenticate with OpenAI API to hide the key "https://techcommunity.microsoft.com/t5/blogs/blogarticleprintpage/blog-id/HealthcareAndLifeSciencesBlog/article-id/1751"

import tiktoken
import streamlit as st
import black
from enum import Enum
import re
import base64
import openai


# Authenticate with OpenAI API
openai.api_type = "azure"
openai.api_base = 'https://xxxxxxx.openai.azure.com/'
openai.api_version = "2023-03-15-preview"
#openai.api_key = 'xxxxxxxxxxxxxxxxxxxxx'
openai.api_key = st.secrets['path']
#model_engine = "text-davinci-002"




class GPTModel(Enum):
    GPT3 = "gpt-3.5-turbo"
    DAVINCI = "text-davinci-002"
    BABBAGE = "text-babbage-002"
    ADA = "text-ada-002"
    CODEX = "text-codex-002"
    GPT4 = "gpt-4.0-turbo"

    def get_token_limit(self):
        # Set the token limit for each GPT model
        token_limits = {
            GPTModel.GPT3: 4096,
            GPTModel.DAVINCI: 4096,
            GPTModel.BABBAGE: 4096,
            GPTModel.ADA: 4096,
            GPTModel.CODEX: 4096,
            GPTModel.GPT4: 4096
        }
        return token_limits.get(self)

def count_metrics(text):
    # Remove excessive newlines and consecutive whitespaces
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Calculate the number of words, characters, and tokens
    word_count = len(text.split())
    char_count = len(text)
    token_count = len(text.split())  # Assuming each word is a token

    return text, word_count, char_count, token_count

def optimize_code(code_input):
    # Format the code using Black
    formatted_code = black.format_str(code_input, mode=black.FileMode())

    # Remove excessive newlines and consecutive whitespaces
    code_input = re.sub(r"\n{2,}", "\n", code_input)
    code_input = re.sub(r"\s{2,}", " ", code_input)

    return formatted_code

def download_file(content, filename):
    b64_content = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64_content}" download="{filename}">Download {filename}</a>'
    return href

def app():
    st.title("Token Metrics and Code Optimizer")
    text_input = st.text_area("Paste your text or code here:")
    selected_model = st.selectbox("Select Model", [model.name for model in GPTModel])

    if st.button("Count Token Metrics"):
        processed_text, word_count, char_count, token_count = count_metrics(text_input)
        st.markdown(f"Word Count: {word_count}")
        st.markdown(f"Character Count: {char_count}")
        st.markdown(f"Token Count: {token_count}")
# Check token count against the token limit for the selected model
        selected_model = GPTModel[selected_model]
        token_limit = selected_model.get_token_limit()
        if token_limit is not None and token_count > token_limit:
            st.warning(f"Token count exceeds the maximum allowed limit for the {selected_model.value} model: {token_limit}")

 
        # Download processed text as a file
        download_link = download_file(processed_text, "processed_text.txt")
        st.markdown(download_link, unsafe_allow_html=True)

    if st.button("Optimize Code"):
        try:
            optimized_code = optimize_code(text_input)
            st.code(optimized_code)

            # Display the counts for the optimized code
            optimized_text, optimized_word_count, optimized_char_count, optimized_token_count = count_metrics(optimized_code)
            st.markdown("Optimized Code Metrics:")
            st.markdown(f"Word Count: {optimized_word_count}")
            st.markdown(f"Character Count: {optimized_char_count}")
            st.markdown(f"Token Count: {optimized_token_count}")

                        # Check token count against the token limit for the selected model
            selected_model = GPTModel[selected_model]
            token_limit = selected_model.get_token_limit()
            if token_limit is not None and optimized_token_count > token_limit:
                st.warning(f"Token count exceeds the maximum allowed limit for the {selected_model.value} model: {token_limit}")

            # Download optimized code as a file
            download_link = download_file(optimized_code, "optimized_code.py")
            st.markdown(download_link, unsafe_allow_html=True)
        except black.InvalidInput as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    app()

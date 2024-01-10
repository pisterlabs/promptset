import streamlit as st
import openai

# Set the GPT-3 API key
openai.api_key = st.secrets["pass"]
st.set_page_config(layout="wide")

# Read the text of the article from a file
# with open("article.txt", "r") as f:
#     article_text = f.read()
st.title("GPT-3 Powered AI Summarizer")
expander_bar = st.expander("About GPT-3")
expander_bar.markdown("""
* **GPT-3** , or the third-generation Generative Pre-trained Transformer, is a neural network machine learning model trained using internet data to generate any type of text. Developed by OpenAI, it requires a small amount of input text to generate large volumes of relevant and sophisticated machine-generated text.
""")
st.subheader('hello, How can I help you today')
article_text = st.text_area("**Enter your scientific texts to summarize**")
output_size = st.radio(label = "What kind of output do you want?", 
                    options= ["To-The-Point", "Concise", "Detailed"])

if output_size == "To-The-Point":
    out_token = 50
elif output_size == "Concise":
    out_token = 128
else:
    out_token = 516

if len(article_text)>100:
    if st.button("Generate Summary",type='primary'):
    # Use GPT-3 to generate a summary of the article
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Please summarize this scientific article for me in a few sentences: " + article_text,
            max_tokens = out_token,
            temperature = 0.5,
        )
        # Print the generated summary
        res = response["choices"][0]["text"]
        st.success(res)
        st.download_button('Download result', res)
else:
    st.warning("Not enough words to summarize!")
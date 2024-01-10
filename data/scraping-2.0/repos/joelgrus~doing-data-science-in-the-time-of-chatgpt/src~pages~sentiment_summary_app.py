import streamlit as st
import openai
import os

# Replace 'your_openai_api_key' with your actual OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]


def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or use the latest model available
        prompt=f"What is the sentiment of this text? {text}",
        max_tokens=60,
        temperature=0.3,
    )
    return response.choices[0].text.strip()


def summarize_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or use the latest model available
        prompt=f"Provide a summary for the following text:\n\n{text}",
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].text.strip()


# Streamlit UI
st.title("Text Analysis Dashboard")

with st.form("text_analysis_form"):
    text_input = st.text_area("Enter the text you'd like to analyze:", height=200)
    submit_button = st.form_submit_button("Analyze Text")

if submit_button and text_input:
    with st.spinner("Analyzing..."):
        sentiment = analyze_sentiment(text_input)
        summary = summarize_text(text_input)

    st.subheader("Sentiment Analysis")
    st.write(sentiment)

    st.subheader("Text Summary")
    st.write(summary)
else:
    st.write("Enter some text and click analyze to get started!")

# To run the Streamlit app, save this code in a file named app.py and run `streamlit run app.py`

import openai
import streamlit as st

st.title("OpenAI Demo")
analyze_button = st.Button("Analyze Text")
openai.api_key = openai.API_KEY
if analyze_button:
    messages = [
        {"role": "system", "content": """You are a helpful sentiment analysis assistant.
            You always respond with the sentiment of the text you are given and the confidence of your sentiment analysis with a number between 0 and 1"""},
        {"role": "user", 
    "content": f"Sentiment analysis of the following text: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    sentiment = response.choices[0].message['content'].strip()
    st.write(sentiment)


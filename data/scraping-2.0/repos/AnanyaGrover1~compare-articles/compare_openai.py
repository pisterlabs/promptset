import streamlit as st
import openai

from dotenv import load_dotenv

load_dotenv()

import os

# Replace 'YOUR_API_KEY' with your actual API key
api_key = os.getenv("OPENAI_API_KEY")

st.title("Compare news articles")


# Accept user input for the URLs
url1 = st.text_input("Enter the first article's URL: ")
url2 = st.text_input("Enter the second article's URL: ")


if st.button("Compare Articles"):
    if url1 and url2:
        # Create the user message for ChatGPT
        user_message = f"""Read these two articles about the same news event (access and read the URLs provided). Summarize the articles together in 3 sets of bullet points:
        * Points of agreement between first article and second article
        * Points of factual disagreement, if any
        * Differences in framing and viewpoint, and selective omissions:\n1. {url1}\n2. {url2}\n\nComparison: """

        # Call the ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            api_key=api_key,
        )

        # Extract the "content" part under "choices" in the response
        comparison_paragraph = response.choices[0].message['content'].strip()

        # Display the comparison paragraph
        st.subheader("Comparison:")
        st.write(comparison_paragraph)
    else:
        st.warning("Please enter both URLs to compare.")

import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

# Initialize the OpenAI client
openai.api_key = user_api_key
prompt = """Act as an AI assistant to summarize news articles.
            You will receive a data of the news and you will 
            summarize it using vocabulary suitable for a high 
            schooler.
        """ 

# set the title of the app
st.title("Sum Mama Sir")
# set the subtitle of the app
st.subheader("Summarize news articles using OpenAI's GPT-3.5-turbo.")
# set input text
input_text = st.text_input("Enter news article URL")

# set the parameters for the API
params = {
    "prompt": prompt,
    "max_tokens": 64,
    "temperature": 0.5,
    "top_p": 1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop": ["\n", " Article:"]
}

# if the user has entered a URL and clicked the button
if st.button("Summarize"):
    # get the URL
    url = input_text
    # get the HTML from the URL
    html = requests.get(url).text
    # parse the HTML using Beautiful Soup
    soup = BeautifulSoup(html, "html.parser")
    # get the text from the HTML
    text = soup.get_text()
    # update the prompt in params
    params["prompt"] = f"""Summarize the following news article:
                        {text}
                        Summarize the article using vocabulary suitable for a high schooler.
                    """
    # get the completion from the API
    completion = openai.Completion.create(**params)
    # set the summary
    summary = completion.choices[0].text
    # display the summary
    st.write(summary)

try:
    completion = openai.Completion.create(**params)
    summary = completion.choices[0].text
    st.write(summary)
except Exception as e:
    st.error(f"Error during API request: {e}")

print("Params:", params)
completion = openai.Completion.create(**params)

print("API Response:", completion)


import streamlit as st
import os
import openai
import requests
from auth import check_password
from langchain import PromptTemplate, LLMChain, OpenAI


def getMad(message, mood):
    # create llm to choose best articles
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    You are a very dumb person and you feel {mood}, you find the  dumbest things to say about any topic;
    {message}
    Above is the topic of interest.
    Please write around 4 tweets about the topic, 
    1/ In the thread, you are feeling {mood}
    2/ Each tweet in the thread is at most 280 characters long;
    3/ The thread needs to address the {message} topic very well
    4/ The thread needs to be  wrong and contradict itself
    5/ The thread should encapsulte a {mood} tone
    6/The thread needs to use complex words in a wrong way
    7/ Limit Response to 1000 tokens
    8/ Start a new line for each tweet
    9/ Use emojis to make the thread more engaging, but not too many

    """

    prompt_template = PromptTemplate(
        input_variables=["message", "mood"], template=template
    )

    article_picker_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    urls = article_picker_chain.predict(message=message, mood=mood)
    return urls


print("password correct")
st.set_page_config("Funny Tweet Maker")

st.header("Funny Tweet Generator")
mood = st.radio(
    "What Mood Am I In?",
    (
        "Happy",
        "Mad",
        "Sad",
        "Hopelessly Romantic",
        "Nerdy",
        "Annoying",
        "Emo",
        "Rhyming",
        "Sarcastic",
        "Poetic",
        "With puns",
        "Cringy",
    ),
)

st.write("Write Any Topic For me to be", mood, "about")

input = st.text_input("Enter Topic")
openAIKey = st.text_input("Enter your OpenAI API Key")
button = st.button("Generate")
if button:
    openai.api_key = openAIKey
    urls = getMad(input, mood)
    st.write(urls)

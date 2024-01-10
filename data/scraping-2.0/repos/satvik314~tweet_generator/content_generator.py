import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
load_dotenv()
import os

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

content_template = """
Give me full {number} posts on {topic} which I can post on {platform}.
"""

content_prompt = PromptTemplate(template = content_template, input_variables = ['number', 'topic', 'platform'])

# gpt3_model = OpenAI(model = "text-davinci-003", temperature = 1)
gpt3_model = ChatOpenAI(temperature=0.5)

content_generator = LLMChain(prompt = content_prompt, llm = gpt3_model)

st.title("Social Media Content Generator 🐦")
st.subheader("🚀 Generate impact social media posts on the go!")

topic = st.text_input("Topic")

platform = st.selectbox("Platform", ["Twitter", "LinkedIn", "WhatsApp"])

number = st.number_input("Number of posts", min_value = 1, max_value = 10, value = 1, step = 1)

if st.button("Generate"):
    posts = content_generator.run(number = number, topic = topic, platform = platform)
    st.write(posts)
    # for tweet in tweets:
    #     st.write(tweet)

import streamlit as st

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

gpt3_model = OpenAI(model_name = 'text-davinci-003')


content_template = """
Give me {number} posts on {topic} which I can post on {platform}.
"""

content_prompt = PromptTemplate(template = content_template, input_variables = ['number', 'topic', 'platform'])

content_chain = LLMChain(
                        prompt = content_prompt,
                        llm = gpt3_model
)


st.title("Social Media Content Generator 🐦")

st.subheader("🚀 Generate impactful social media posts on the go!")

topic = st.text_input("Topic")

platform = st.selectbox(
    'Platform',
    ('Twitter', 'LinkedIn', 'WhatsApp'))

# number = st.number_input('Insert a number')
number = st.number_input("Number of posts", min_value = 1, max_value = 10, value = 1, step = 1)

st.button("Generate")
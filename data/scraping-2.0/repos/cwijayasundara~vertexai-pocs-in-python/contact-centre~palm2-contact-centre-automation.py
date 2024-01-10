import streamlit as st
import vertexai

from datasets import load_dataset
from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.chains import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from PIL import Image

PROJECT_ID = "ibm-keras"
REGION = "us-central1"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)
vertexai.init(
    project=PROJECT_ID,
    location=REGION
)

ignore_warnings = True

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=False,
    ignore_warnings=True
)

# load the dataset from huggingface
dataset = load_dataset("banking77")
# Sort the dataset by the length of the customer texts
sorted_data = sorted(dataset['train'], key=lambda x: len(x['text']), reverse=True)
longest_ten_texts = [entry["text"] for entry in sorted_data[:10]]

# SequentialChain
english_translator_prompt = ChatPromptTemplate.from_template(
    "Translate the following enquiry to english:{Review}")

# chain 1: input= Review and output= English_Review
english_translator_chain = LLMChain(llm=llm, prompt=english_translator_prompt, output_key="English_Review")

# summary chain
summary_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following enquiry in no longer than 100 words?: {English_Review}")

# chain 2: input= English_Review and output= summary
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# sentiment chain
sentiment_prompt = ChatPromptTemplate.from_template("Identify the sentiment of the the following enquiry in single "
                                                    "word, positive, negative or neutral: {summary}")

sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment")

# Intent chain
intent_prompt = ChatPromptTemplate.from_template("Identify the intent of the the following enquiry in single sentence"
                                                 "\n\n{summary}"
                                                 )
intent_chain = LLMChain(llm=llm, prompt=intent_prompt, output_key="intent")

# Identity the original language the enquiry was written in
language_prompt = ChatPromptTemplate.from_template("What language is the following enquiry:\n\n{Review}")

# input= Review and output= language
language_chain = LLMChain(llm=llm, prompt=language_prompt, output_key="language")

# prompt template 4: follow-up message
response_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response in very professionally to the following "
    "summary and sentiment in the specified language:"
    "\n\nSummary: {summary}\n\nsentiment: {sentiment}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="followup_message")

# overall_chain: input= Review
# and output= English_Review,summary, follow up_message
overall_chain = SequentialChain(
    chains=[english_translator_chain, summary_chain, sentiment_chain, intent_chain, language_chain, response_chain],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "sentiment", "intent", "language", "followup_message"],
    verbose=True
)


def display_row(customer_enquery):
    return overall_chain(customer_enquery)

# page construction
st.set_page_config(page_title="Contact Centre Automation for ABC Plc", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="robo.png")

icon = Image.open("../trader-dashboard/robo.png")
st.image(icon, width=100)

st.title("Contact Centre Automation for ABC Plc")
# Generate a dropdown with options from longest_ten_texts
selected_enquery = st.selectbox('Select an Enquery', longest_ten_texts)

# Call the display_row function and print the result
result = display_row(selected_enquery)
st.write(result)

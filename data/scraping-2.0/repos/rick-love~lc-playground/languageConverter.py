from langchain.llms import OpenAI
import streamlit as st
from config import get_OpenAI
from langchain.prompts import PromptTemplate

template = """
Below is a text message that maybe poorly written.
Your goal is to:
- Properly format the text
- Convert the text to the desired tone
- Convert the text to the desired language

Here is an example of different tones:
- Formal: "I am writing to inform you that the product you ordered has been shipped."
- Informal: "Hey! Just wanted to let you know that your order has been shipped."
- Casual: "Yo! Your order has shipped."

Here are some examples of different languages:
- English: "I am writing to inform you that the product you ordered has been shipped."
- German: "Ich schreibe Ihnen, um Sie darüber zu informieren, dass das von Ihnen bestellte Produkt versandt wurde."
- Spanish: "Le escribo para informarle que el producto que ha pedido ha sido enviado."

Below is the text, tone and language:
TONE: {tone}
LANGUAGE: {language}
TEXT: {text}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=['tone', 'language', 'text'],
    template=template,
)
# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")

def load_LLM():
    llm = OpenAI(temperature=0, max_tokens=100)
    return llm

llm = load_LLM()



st.set_page_config(page_title="Langchain Playground", page_icon="🧊", layout="wide")
st.header("Langchain Playground")

col1, col2, col3 = st.columns(3)


with col1:
    st.write('Creating a language conversion app which will be the base for the Chat Agent.')
with col2:
    st.image('day_1.png')
    
st.markdown('---')
st.markdown('## Enter your text here:')

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox('Tone', ('Formal', 'Informal', 'Casual'))
with col2:
    option_lang = st.selectbox('Language', ('English', 'German', 'Spanish'))


def get_text():
    input_text = st.text_area(label='text', label_visibility='hidden', placeholder='Enter text here', key='text_area')
    return input_text

text_input = get_text()

st.markdown('Converted text:')

if text_input:
    prompt_with_text = prompt.format(
        tone=option_tone,
        language=option_lang,
        text=text_input
    )
    formatted_text = llm(prompt_with_text)
    st.write(formatted_text)
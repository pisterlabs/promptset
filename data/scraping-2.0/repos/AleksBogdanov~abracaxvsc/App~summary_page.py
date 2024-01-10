import os
import cohere  # Loading Cohere Libray
import streamlit as st
from dotenv import load_dotenv
from utils import pdf_to_string

# The summarization is created by calling the API from Cohere. Cohere is LLM (Large Language Model)
# It can summarize lengthy documents up to 50,000 characters, which is equivalent to 18-20 pages in
# a single-spaced format. How does it work? It creates embeddings of the text, which are representation
# of the text in the form of numbers. There numbers can easily be grouped by similarity and in this way
# one can understand which part of the text speak about the same thing.

load_dotenv()  # Loading the API keys safely
st.set_page_config(page_title="Резюме>")
st.subheader("")
st.markdown(
    "<h1 style='text-align: center; color: black;'>Създайте кратко резюме на документ по Ваш избор </h1>",
    unsafe_allow_html=True)  # Name displayed on the page

hide_button_style = """
<style>
  .css-14xtw13 {    
    display: none;
  }
</style>

"""
st.markdown(hide_button_style, unsafe_allow_html=True)

def summarize(
        document: str,
        summary_length: str,
        summary_format: str,
        extractiveness: str = "high",
        temperature: float = 0.6,
) -> str:
    """
    Generates a summary for the input document using Cohere's summarize API.
        Args:
            document (`str`):
                The document given by the user for which summary must be generated.
            summary_length (`str`):
                A value such as 'short', 'medium', 'long' indicating the length of the summary.
            summary_format (`str`):
                This indicates whether the generated summary should be in 'paragraph' format or 'bullets'.
            extractiveness (`str`, *optional*, defaults to 'high'):
                A value such as 'low', 'medium', 'high' indicating how close the generated summary should be in meaning to the original text.
            temperature (`str`):
                This controls the randomness of the output. Lower values tend to generate more “predictable” output, while higher values tend to generate more “creative” output.
        Returns:
            generated_summary (`str`):
                The generated summary from the summarization model.
    """

    summary_response = cohere.Client(os.getenv('COHERE_API_KEY')).summarize(
        text=document,
        length=summary_length,
        format=summary_format,
        model="summarize-xlarge",
        extractiveness=extractiveness,
        temperature=temperature,
    )
    generated_summary = summary_response.summary
    return generated_summary


uploaded_file = st.file_uploader("Изберете файл", type="pdf")  # Uploading the file
if uploaded_file is not None:  # Check whether file is uploaded
    st.cache_resource.clear()
    text = pdf_to_string(uploaded_file)  # Convert the pdf to string
    summary_format_btn = st.radio(
        "Изберете типа на резюмето",
        ('Булети', 'Параграф'))
    document_content = st.empty()
    temperature = st.slider(
        'Изберете температура (Мярка за непредвидимост и креативност на модела: 0 е изцяло детерминистичен, 5'
        ' е максимално волатилен. Оптима    лно е между 0.5 и 1)', min_value=0.0, step=0.01, max_value=5.0, value=1.0)
    # Some CSS to make buttons more visually appealing
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ce1126;
        color: white;
        height: 3em;
        width: 12em;
        border-radius:10px;
        border:3px solid #000000;
        font-size:20px;
        font-weight: bold;
        margin: auto;
        display: block;
    }
    
    div.stButton > button:hover {
        background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
        background-color:#ce1126;
    }
    
    div.stButton > button:active {
        position:relative;
        top:3px;
    }
    </style>""", unsafe_allow_html=True)
    # Create Summary button
    summary_button = st.button("Генерирай Резюме")
    bul_to_english_summary_format = {'Булети': 'bullets',
                                     'Параграф': 'paragraph'}  # To store translatiion from BG to EN

    # Generate Summary
    summary_text = summarize(text, summary_length='long',
                             summary_format=bul_to_english_summary_format[summary_format_btn],
                             extractiveness='high',
                             temperature=temperature)

    # Calculate the character count using the function
    character_count = len(summary_text)
    # Adjust the size of the text area based on the character count
    if summary_format_btn == 'Булети':
        adjust_height = int(character_count // 2.6)  # Adjust the factor
    else:
        adjust_height = (character_count // 3)  # Adjust the factor

    # If the summary button is pressed, the text appears in the text area
    if summary_button:
        txt = st.text_area('Резюме', summary_text, height=adjust_height)

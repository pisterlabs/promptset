import streamlit as st
import pytesseract
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from deep_translator import GoogleTranslator
import openai
import os
import logging
from PyPDF2 import PdfReader
from multi_lang import LANGUAGES

st.set_page_config(
        page_title="Primo JAG",
        page_icon='🤖',
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'mailto:jag.solutionshub@gmail.com',
            'Report a bug': "mailto:jag.solutionshub@gmail.com",
            'About': "Welcome to your AI cusin!. \
                Translation: This feature allows users to translate documents from English to either Spanish or English using Google's Translation API.\
                    Document Extraction: The script uses the Pytesseract library to extract text from image files or photos \
                    taken by a camera. If the user uploads a PDF file, the script uses PyPDF2 to extract the text.\
                    Document Analysis: The OpenAI's GPT-3.5-turbo model is used to analyze the translated document \
                    and provide a summarized and easy-to-understand output, which includes the most important points,\
                    a concise summary of the main ideas, essential facts, and suggestions on the next steps to take based\
                    on the content of the document.Streamlit Interface: The entire application is built using Streamlit, \
                    which provides an intuitive and user-friendly interface for users to upload their documents,\
                    select their language, and see the translated and summarized content of their documents.Logging: \
                    The script uses Python's built-in logging module to log errors that occur during the execution of the \
                    application."
        }
    )

def setup_logging():
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    
def extract_text_from_image(image,lang):
    """Extract text from a given image."""
    try:
        with st.spinner(LANGUAGES[lang]['ext_img']):
            extracted_text = pytesseract.image_to_string(image).strip()
            if not extracted_text:
                st.warning(LANGUAGES[lang]['error_ext_img'])
                return None
            return extracted_text
    except pytesseract.pytesseract.TesseractNotFoundError as e:
        logging.error(f'Tesseract not found. Please check the installation: {e}')
        return None
    except Exception as e:
        logging.error(f'Error during text extraction: {e}')
        return None

def extract_text_from_pdf(pdf_file,lang):
    """Extract text from a given PDF file."""
    try:
        with st.spinner(LANGUAGES[lang]['ext']):
            pdf = PdfReader(pdf_file)
            total_pages = len(pdf.pages)
            extracted_text = ""
            for page_number in range(total_pages):
                page = pdf.pages[page_number]
                extracted_text += page.extract_text()
            if not extracted_text:
                st.warning(LANGUAGES[lang]['error_ext'])
                return None
            return extracted_text
    except Exception as e:
        logging.error(f'Error during translation: {e}')
        return None

def translate_text(text, lang):
    """Translate a given text to Spanish."""
    try:
        with st.spinner(LANGUAGES[lang]['language']):
            translated_text = GoogleTranslator(source='auto', target=lang).translate(text)
            return translated_text
    except Exception as e:
        logging.error(f'Error during translation: {e}')
        return None

def loadOpenAI():
    try:
        OPEN_AI = os.environ['OPEN_AI']
        openai.api_key = OPEN_AI
        AI_key_message = "Open AI key was accepted!"
    except KeyError:
        OPEN_AI = "OPEN_AI Token not available!"
        AI_key_message = "GetResponse key was not avaialble!"
    print(AI_key_message)

def AI_text_OpenAI(text, document_type, target_language):
    AI_personality = "You are a highly skilled language assistant who is my cousin born in cuba so I want you to be very cuban in your response and greet me as you cousin using extremely cuban lingo when assisting. Your capabilities include translating text in, \
        highlighting key points, summarizing content, identifying need-to-know facts, and providing actionable \
        recommendations based on the text provided."

    document_type_specific_instructions = {
        "es":"Cuban Spanish",
        "immigration": "This document relates to immigration matters. Please pay special attention to details related to case status, dates, deadlines, visa requirements, and any supporting documentation.",
        "legal": "This is a legal document. Please emphasize legal terms, conditions, obligations, rights, any penalties or consequences, and the overall legal implications.",
        "doctor": "This document is from a medical professional. Please focus on medical terms, diagnoses, treatment plans, medications, and any medical recommendations or precautions.",
        "personal": "This is a personal document. Please respect the privacy and sensitivity of the content. Highlight any significant personal events, dates, important contacts, or other pertinent information.",
        "en":"English",
        "inmigración":"This document relates to immigration matters. Please pay special attention to details related to case status, dates, deadlines, visa requirements, and any supporting documentation.",
        "legal": "This is a legal document. Please emphasize legal terms, conditions, obligations, rights, any penalties or consequences, and the overall legal implications.",
        'médico': "This document is from a medical professional. Please focus on medical terms, diagnoses, treatment plans, medications, and any medical recommendations or precautions.",
        "personal": "This is a personal document. Please respect the privacy and sensitivity of the content. Highlight any significant personal events, dates, important contacts, or other pertinent information."
}
    language = document_type_specific_instructions.get(target_language.lower(), "Cuban Spanish")
    specific_instruction = document_type_specific_instructions.get(document_type.lower(), "")
    model_id = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(
        model = model_id,
        messages = [
            {"role": "system", "content": AI_personality},
            {"role": "user", "content": f"I have a {document_type} document that I need your assistance with. \
                {specific_instruction} Could you please do the following: \n\
                    \n1. Highlight the most important points using easy-to-understand language.\
                    \n2. Summarize the main ideas in a concise and straightforward manner, avoiding complex terms and jargon.\
                    \n3. Provide the essential 'need-to-know' facts using plain and accessible language.\
                    \n4. Based on the document's content, offer actionable comments or suggestions using simple words on what steps should be taken next. \
                    \n\n Make sure you response is in {language} using cuban expression and vocabulary and dont forget you are my cousin.The document text is as follows: {text}"},
        ],
        temperature = 0.2,
        max_tokens =  500,
    )    
    return response.choices[0].message.content


def main():
    loadOpenAI()
    
    with st.sidebar:
        st.header("Please select your language")
        lang = st.sidebar.radio(':flag-cu: - :flag-us:', ('es', 'en'),horizontal=True, index=0)
        img = Image.open('img/logo.png')
        st.image(img)
        st.markdown(LANGUAGES[lang]['sidebar_content'])
        
    st.title('Tu Primo JAG - "El Gringo" AI')
    
    radio_option = st.radio(LANGUAGES[lang]['document_format'], LANGUAGES[lang]['doc_options'],horizontal=True,index=len(LANGUAGES[lang]['doc_options'])-1)
    
    if radio_option in ("Image","Imagen"):
        uploaded_img = st.file_uploader(LANGUAGES[lang]['upload_img'], type=["jpg", "png", "jpeg"])
        if uploaded_img is not None:
            try:
                img = Image.open(uploaded_img)
            except IOError:
                st.error(LANGUAGES[lang]["img_error"])
                return  # If the file couldn't be read as an image, stop here
            st.image(img, caption='Uploaded Image', use_column_width='auto')
            type_of_doc = st.selectbox(
            LANGUAGES[lang]['document_type'],
            (LANGUAGES[lang]['doc_type_options']), index=0)
            if st.button(LANGUAGES[lang]['explain_button']):
                try:
                    extracted_text = extract_text_from_image(img,lang)
                    if not extracted_text:
                        return
                    translated_text = translate_text(extracted_text,lang)
                except Exception as e:
                    logging.error(f'f"An error occurred: {str(e)}')
                else:
                    st.success(LANGUAGES[lang]['translation_successful'])
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander(LANGUAGES[lang]['translated_text']):
                            st.write(translated_text)
                    with col2:
                        with st.spinner(LANGUAGES[lang]['primo_analyzing']):
                            response = AI_text_OpenAI(extracted_text, type_of_doc, lang)
                        st.snow()
                        st.write(response)
                        
    elif radio_option in ("Camera","Cámara"):
        st.warning(LANGUAGES[lang]['cam_warning'])
        img_file_buffer = st.camera_input("")
        if img_file_buffer is not None:
            try:
                img = Image.open(img_file_buffer)
            except IOError:
                st.error(LANGUAGES[lang]["error_img"])
                return  # If the image couldn't be read, stop here
            st.image(img, caption='Took Image', use_column_width=True)
            type_of_doc = st.selectbox(
            LANGUAGES[lang]['document_type'],
            (LANGUAGES[lang]['doc_type_options']), index=0)
            if st.button(LANGUAGES[lang]['explain_button']):
                try:
                    extracted_text = extract_text_from_image(img,lang)
                    if not extracted_text:
                        return
                    translated_text = translate_text(extracted_text,lang)
                except Exception as e:
                    logging.error(f'f"An error occurred: {str(e)}')
                else:
                    if translated_text is not None:
                        st.success(LANGUAGES[lang]['translation_successful'])
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander(LANGUAGES[lang]['translated_text']):
                                st.write(translated_text)
                        with col2:
                            with st.spinner(LANGUAGES[lang]['primo_analyzing']):
                                response = AI_text_OpenAI(extracted_text, type_of_doc, lang)
                            st.snow()
                            st.write(response)
                    else:
                        st.warning("The translation could not be completed.")
                 
    elif radio_option == "PDF":
        uploaded_pdf = st.file_uploader(LANGUAGES[lang]['upload_pdf'], type=["pdf"])        
        if uploaded_pdf is not None:
            type_of_doc = st.selectbox(
            LANGUAGES[lang]['document_type'],
            (LANGUAGES[lang]['doc_type_options']), index=0)
            if st.button(LANGUAGES[lang]['explain_button']):
                try:
                    extracted_text = extract_text_from_pdf(uploaded_pdf,lang)
                    if not extracted_text:
                        return
                    elif len(extracted_text) > 5000:
                        st.error(LANGUAGES[lang]['long_text'])
                        return
                    
                    translated_text = translate_text(extracted_text,lang)
                except Exception as e:
                    logging.error(f'f"An error occurred: {str(e)}')
                else:
                    if translated_text is not None:
                        st.success(LANGUAGES[lang]['translation_successful'])
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander(LANGUAGES[lang]['translated_text']):
                                st.write(translated_text)
                        with col2:
                            with st.spinner(LANGUAGES[lang]['primo_analyzing']):
                                response = AI_text_OpenAI(extracted_text, type_of_doc, lang)
                            st.snow()
                            st.write(response)

if __name__ == '__main__':
    main()

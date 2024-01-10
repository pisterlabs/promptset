import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import google.generativeai as genai
from streamlit_extras.add_vertical_space import add_vertical_space
import PIL.Image

# Import Translator 
from googletrans import Translator

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Sidebar contents
with st.sidebar:
    st.title('🦜️🔗VK - ALL IN ONE CHATBOT')
    st.markdown('''
    ## About APP:

    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Gemini](https://ai.google.dev/tutorials/python_quickstart#chat_conversations)
    - [Palm2](https://ai.google/discover/palm2)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/venkat-vk/)
    
    ''')

    add_vertical_space(1)
    st.write('💡All about Gemini and Palm2 exploration...., created by VK🤗')
    st.write('💡Full credit: Gemini AI🤗')

st.title("VK - Vision,Multiple Pdf & Prompt Based Chatbot With Translator")

# Multiple PDF-based chatbot
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)

selected_option = st.selectbox("Select Processing Type", ["Multiple PDFs", "Vision", "Prompt"])

def clear_session_state():
    st.session_state.conversation = None
    st.session_state.chatHistory = None

# Clear session state if the user changes the processing type
if st.session_state.get('selected_option') != selected_option:
    clear_session_state()

# Store the current processing type in session state
st.session_state.selected_option = selected_option

# Model selection for Vision-based processing
models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
selected_model = st.selectbox("Select a Generative Model for Vision", models)

# Handle selected processing type
if selected_option == "Multiple PDFs":

    st.title("Multiple PDF Bot")

    # Multiple PDF-based chatbot
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("PDFs Processed")

    # User input for PDF-based chatbot
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    # Get the generated content for translation
    response = st.session_state.chatHistory[-1].content

    # Translation
    dest_lang = st.selectbox("Select destination language", ["Hindi", "Tamil", "Telugu", "Punjabi", "German", "English", "French"])
    if dest_lang:
        language_codes = {
            "Hindi": "hi",
            "Tamil": "ta",
            "Telugu": "te",
            "Punjabi": "pa",
            "German": "de",
            "English": "en",
            "French": "fr"
        }
        dest_lang_code = language_codes.get(dest_lang)
        
        try:
            translator = Translator()
            translation = translator.translate(response, src="en", dest=dest_lang_code)
            if translation is not None and hasattr(translation, 'text') and translation.text:
                st.write(f"**Translated Answer ({dest_lang}):** {translation.text}")
            else:
                st.error("Translation failed. Please check your input and try again.")
        except Exception as e:
            st.error(f"An error occurred during translation: {str(e)}")

elif selected_option == "Vision":

    st.title("Vision Bot")

    # File uploader for vision model
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_file is not None:
        # Read the uploaded image
        img = PIL.Image.open(uploaded_file)

        # Display the uploaded image
        st.subheader("Uploaded Image:")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Generate content based on the image
        vision_model = genai.GenerativeModel(selected_model)
        response = vision_model.generate_content(img)

        # Display generated content
        st.subheader("Generated Content:")
        st.write(response.text)
        
        # Get the generated content for translation
        response = response.text

        # Translation step
        dest_lang = st.selectbox("Select destination language", ["Hindi", "Tamil", "Telugu", "Punjabi", "German", "English", "French"])
        if dest_lang:
            language_codes = {
                "Hindi": "hi",
                "Tamil": "ta",
                "Telugu": "te",
                "Punjabi": "pa",
                "German": "de",
                "English": "en",
                "French": "fr"
            }
            dest_lang_code = language_codes.get(dest_lang)
            
            try:
                translator = Translator()
                translation = translator.translate(response, src="en", dest=dest_lang_code)
                if translation is not None and hasattr(translation, 'text') and translation.text:
                    st.write(f"**Translated Answer ({dest_lang}):** {translation.text}")
                else:
                    st.error("Translation failed. Please check your input and try again.")
            except Exception as e:
                st.error(f"An error occurred during translation: {str(e)}")

else:
    st.title("General Prompt Bot")

    # Text input form
    question = st.text_input("Ask a question:")
    submitted = st.button("Submit")

    if submitted:
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content(question)

        # Display question and generated content
        st.subheader("Question:")
        st.write(question)

        st.subheader("Generated Content:")
        st.write(response.text)
        
        # Get the generated content for translation
        response = response.text

        # Translation step
        dest_lang = st.selectbox("Select destination language", ["Hindi", "Tamil", "Telugu", "Punjabi", "German", "English", "French"])
        if dest_lang:
            language_codes = {
                "Hindi": "hi",
                "Tamil": "ta",
                "Telugu": "te",
                "Punjabi": "pa",
                "German": "de",
                "English": "en",
                "French": "fr"
            }
            dest_lang_code = language_codes.get(dest_lang)
            
            try:
                translator = Translator()
                translation = translator.translate(response, src="en", dest=dest_lang_code)
                if translation is not None and hasattr(translation, 'text') and translation.text:
                    st.write(f"**Translated Answer ({dest_lang}):** {translation.text}")
                else:
                    st.error("Translation failed. Please check your input and try again.")
            except Exception as e:
                st.error(f"An error occurred during translation: {str(e)}")

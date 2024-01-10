# import libraries
# The libraries are imported. Streamlit is used to create the web app. LangChain with OpenAI is used for text summarization. Document is used to convert text into a document object. CharacterTextSplitter is used to split the text into smaller segments. Translator is used to translate the text. 
import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from translate import Translator 


# The app is created. st.set_page_config() is used to set the page title.
st.set_page_config(page_title='Text summarizer and translator app 📄🌍', page_icon='🌻')

# Language dictionary mapping codes to full names, instead of using the codes to make it more user-friendly
language_dict = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'nl': 'Dutch',
    'it': 'Italian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic'
}

# Function to perform text summarization
# The function summarize_text() takes the input text and the OpenAI API key as input. The Large Language Model is initialized with OpenAI. The input text is split into smaller segments. This is done to make sure that the text is not too long for the summarization chain. If the text segments are not valid, an error message is shown. The text segments are converted into Document objects. This is done to make sure that the text is in the right format for the summarization chain. If the document list is not valid, an error message is shown. The text summarization chain is set up. The summarization chain is run and the result is returned. 
def summarize_text(input_text, api_key):
    # Initialize the Large Language Model
    model = OpenAI(temperature=0, openai_api_key=api_key)

    # Check if input text is valid
    if not input_text.strip():
        st.error("Please enter some text to summarize.")
        return None

    # Split text into smaller segments
    splitter = CharacterTextSplitter()
    text_segments = splitter.split_text(input_text)

    # Check if text segments are valid
    if not text_segments:
        st.error("Text splitting resulted in no valid segments.")
        return None

    # Convert text segments into Document objects
    document_list = [Document(page_content=segment) for segment in text_segments if segment.strip()]

    # Check if document list is valid
    if not document_list:
        st.error("No valid documents were created from the text segments.")
        return None

    # Set up the text summarization chain
    summarization_chain = load_summarize_chain(model, chain_type='map_reduce')

    # Run the summarization chain and return the result
    summarized_text = summarization_chain.run(document_list)
    return summarized_text

# Function to translate text
# The function translate_text() takes the input text, the source language and the target language as input. Before the user could only input the target language, but then it would only translate if the input was English, now the user can also type in another language. The language code for the languages is found. The Translator is initialized with the target language code. The max_length is set to 500, because the Translator can only translate text with a maximum length of 500 characters. The input text is split into smaller segments. This is done to make sure that the text is not too long for the translation. The text segments are translated and combined. The translated text is returned. 
def translate_text(text, source_language_name, target_language_name):
    # Find the language code for the selected language
    source_language_code = [code for code, name in language_dict.items() if name == source_language_name][0] # the language of the input text
    target_language_code = [code for code, name in language_dict.items() if name == target_language_name][0] # language of the output text
    translator = Translator(from_lang=source_language_code, to_lang=target_language_code) 
    max_length = 500 

    # Text is split into chunks of max_length, so that the text is not too long for the translation. 
    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)] 

    # Each chunk is translated  and combined
    translated_chunks = [translator.translate(chunk) for chunk in text_chunks]
    translated_text = ' '.join(translated_chunks)

    return translated_text

# Home page function
# The function home_page() is used to create the home page. The title and description of the app are shown. The flowchart of the app is shown. 
def home_page():
    st.title("Welcome to the text summarizer and translator app! 📄🌍")
    st.markdown("")
    st.markdown("""
                
                This app was created as part of the NLP assignment. This app allows you to summarize and translate text.
                - **Summarize**: rewrite your text into a shorter version.
                - **Translate**: translate your text into various languages.
                """)
    st.markdown("""In order to use the app, you need to have an OpenAI API key. You can get one <a href="https://beta.openai.com/">here.</a>""", unsafe_allow_html=True)
    st.markdown("""To view the code and the comments, go to the <a href="https://github.com/irisvdboomen/NLP-assignment">GitHub</a> repository.""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
                #### Design challenge statement:
                _"Design an **AI-powered web application** to enable **researchers, students, and professionals** in **multilingual environments** to **quickly summarize and translate documents with high accuracy and efficiency.**"_
    """)
    st.markdown("")
    st.markdown("#### Design process:")
    image_path = "NLP assignment flowchart.png" 
    st.image(image_path, caption='Flowchart of the app')
    st.markdown("")
    st.markdown("Navigate to the 'Summarizer and translator' page from the sidebar to use the app.")

# The function nlp_assignment_page() is used to create the page for the NLP assignment. The title of the page is shown. The user can input the text they want to summarize and translate. The user can input their OpenAI API key. The user can choose between summarizing and translating. If the user chooses to summarize, the text is summarized. If the user chooses to translate, the user can choose the language they want to translate to. The text is then translated. 
def nlp_assignment_page():
    st.header('Text summarizer and translator app 📄🌍')

    # User input for text summarization
    user_input = st.text_area("Paste the text you want to summarize and translate:", height=150)

    # Input for OpenAI API key
    api_key = st.text_input('Enter your OpenAI API Key', type='password', help='Your API key should start with "sk-"')

    # Initialize summarized_text
    summarized_text = ""

    # Radio buttons for choosing action
    action = st.radio("Choose an action:", ('Summarize', 'Translate'))

    # # Process the selected action
    if action == 'Summarize' and api_key.startswith('sk-') and user_input:
        with st.spinner('Summarizing...'):
            summarized_text = summarize_text(user_input, api_key)
            st.subheader('Summarized text:')
            st.write(summarized_text)
    elif action == 'Translate' and api_key.startswith('sk-') and user_input:
        # Dropdown for selecting translation language
        source_language_choice = st.selectbox("Translate from:", list(language_dict.values()))
        target_language_choice = st.selectbox("Translate to:", list(language_dict.values()))
        with st.spinner('Translating...'):
            # Translate the summarized text if available; otherwise, translate the original input
            text_to_translate = summarized_text if summarized_text else user_input
            translated_text = translate_text(text_to_translate, source_language_choice, target_language_choice)
            st.subheader('Translated text:')
            st.write(translated_text)

# The function main() is used to create the main app. The navigation sidebar is created. The user can choose between the home page and the NLP assignment page. A footer is added. 	
# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["About", "Summarizer and translator"])

    if page == "About":
        home_page()
    elif page == "Summarizer and translator":
        nlp_assignment_page()

    st.markdown("---")
    st.markdown("")
    st.markdown('<small>Created by <a href="https://github.com/irisvdboomen/NLP-assignment">Iris van den Boomen</a></small>', unsafe_allow_html=True)   
    st.markdown("")

if __name__ == "__main__":
    main()
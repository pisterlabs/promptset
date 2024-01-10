import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import openai

# Function to generate a response using Langchain and OpenAI
def analyze_chat(chat_text, openai_api_key):
    openai.api_key = openai_api_key
    
    prompt = (
        "Here is a conversation between two people. Analyze this conversation and "
        "give insights into their communication style, tell them their key personal traits, "
        "and provide recommendations to enhance their relationship:\n\n"
        f"{chat_text}"
    )
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Replace with the appropriate engine
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Placeholder function to analyze chat using OpenAI's API
def analyze_chat(chat_text, openai_api_key):
    openai.api_key = openai_api_key
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Replace with the appropriate engine
            prompt="Analyze the following chat and provide insights:\n" + chat_text,
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Main function where the Streamlit app is defined
def main():
    st.set_page_config(page_title='👋 Welcome to the Chat Analyzer with Kaare')
    st.title('👋 Welcome to the Chat Analyzer with Kaare')

    # Intro and security assurance
    st.markdown("""
        ## Introduction
        Kaare AI provides insights into your personal patterns and communication styles by analyzing your chat history by using ChatGPTs APIs. 
        
        ## Security and Privacy
        **Your privacy is very important to us!** We ensure that:
        - All conversations are processed securely.
        - We do not store your conversations.
        - We do not have access to any of the data you upload here.
        
        For any questions or feedback, feel free to reach out at [burcuycareer@gmail.com](mailto:burcuycareer@gmail.com).
        """)

    # WhatsApp Chat Analyzer part
    st.header('WhatsApp Chat Analyzer')
    chat_file = st.file_uploader('Upload WhatsApp chat text file', type='txt')
    openai_api_key = st.text_input('OpenAI API Key for Chat Analysis', type='password')
    if st.button('Analyze Chat'):
        if chat_file is not None and openai_api_key:
            chat_text = chat_file.getvalue().decode("utf-8")
            with st.spinner('Analyzing chat...'):
                analysis_result = analyze_chat(chat_text, openai_api_key)
                st.success('Chat analysis is complete!')
                st.text_area("Analysis Result", analysis_result, height=300)
        else:
            st.error('Please upload a chat file and provide the OpenAI API key.')

    # Part for summarization using Langchain and OpenAI
    st.header('Analyze your Chat with Kaare AI')
    txt_input = st.text_area('Enter your chat to analyze', '', height=200)
    openai_api_key_summ = st.text_input('OpenAI API Key for Summarization', type='password')
    if st.button('Submit for Analyze'):
        if openai_api_key_summ and txt_input:
            with st.spinner('Calculating...'):
                response = generate_response(txt_input, openai_api_key_summ)
                st.success('Analysis complete!')
                st.text_area("Analysis Result", response, height=300)
        else:
            st.error('Please enter text to analyze and provide the OpenAI API key.')

# Run the main function
if __name__ == "__main__":
    main()
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import requests
from bs4 import BeautifulSoup

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Create a radio button to choose the chat mode
chat_mode = st.radio("Select Chat Mode:", ("webscrap Chat", "youtube transcript"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to scrape a website and return its content

def webscrap(name):
    # Replace this URL with the one you want to scrape
    url = f'https://www.{name}.com'

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        return page_text
    else:
        return None

def main():
    st.title("Web Scraper Chatbot")
    st.header("Web Scraper Chatbot")

    name = st.text_input("Enter the website name to scrape")

    if st.button("Scrape and Analyze"):
        # Call the webscrap function to scrape the website and save the content
        content = webscrap(name)

        if content is not None:
            st.write("Content has been scraped and saved.")
            max_length = 1800
            original_string = content
            strings_list = []

            while len(original_string) > max_length:
                strings_list.append(original_string[:max_length])
                original_string = original_string[max_length:]
            strings_list.append(original_string)

            # Create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(strings_list, embedding=embeddings)

            prompt = st.text_area("Ask a question:")
            
            if st.button("Ask"):
                user_question = prompt
                print("yha hu")
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    print("context")
                    llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9)

                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()


import os
import openai
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from pages.Welcome import user_input, onboard
from functions.chains import search, find_best_article_urls, get_content_from_urls, summarize, generate_list

def main():
    load_dotenv(find_dotenv())

    st.set_page_config(page_title="Lemo - To-Do Action Items ", page_icon=":judge:", layout="wide", initial_sidebar_state="collapsed",)

    st.header("Lemo - Your AI Paralegal  :judge:")
    openaiapi = os.getenv("OPENAI_API_KEY")
    query = st.text_input("Topic of client and attorney To Do list")
    # Create a side bar for Uploading Documents
    st.sidebar.header("Upload Documents")
    # add input for uploading documents
    uploaded_files = st.sidebar.file_uploader("Upload your files here", accept_multiple_files=True)
    # Create a function to store files
    def file_store():
        for uploaded_file in uploaded_files: # type: ignore
            with open(os.path.join("uploaded_files", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("Saved File: {} to uploaded_files".format(uploaded_file.name))
    # Create a button to store files
    if st.sidebar.button("Save Files"):
        file_store()
    openai.api_key = openaiapi


    if query:
        print(query)
        st.write("Generating client and attorney To Do list for: ", query)
        
        search_results = search(query)
        urls = find_best_article_urls(search_results, query)
        data = get_content_from_urls(urls)
        summaries = summarize(data, query)
        list = generate_list(summaries, query)

        with st.expander("search results"):
            st.info(search_results)
        with st.expander("best urls"):
            st.info(urls)
        with st.expander("data"):
            st.info(data)
        with st.expander("summaries"):
            st.info(summaries)
        with st.expander("list"):
            st.info(list)


if __name__ == '__main__':
    main()
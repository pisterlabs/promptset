# Library
import openai
import streamlit as st
import pandas as pd
from datetime import datetime
import time
from RAGBot_functions import create_qa_chain, answer_default, answer_from_doc, answer_idk_ext_context, \
    is_yes_no, is_same_page, has_page_numbers, is_request, extract_core_request, get_last_question, \
    answer_what_else
from utils import read_pdf, process_docs, get_pdf_len, search_doc_from_knowledge_base, \
    str_to_list, get_doc_pages

# Custom Streamlit app title and icon
st.set_page_config(
    page_title="ChatPDF",
    page_icon=":robot_face:",
)

# Set the title
st.title("ChatPDF")

# Sidebar Configuration
st.sidebar.title("Model Configuration")

# Model Name Selector
model_name = st.sidebar.selectbox(
    "Select a Model",
    ["gpt-3.5-turbo", ],  # Add more model names as needed
    key="model_name",
)

# Check box, external knowledge
external = st.sidebar.checkbox('Using external knowledge')


# Check box, using the same page
same_page = st.sidebar.checkbox('Query in the same page')


# Upload pdf
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None and "knowledge_base" not in st.session_state:
    # Process the uploaded PDF file
    pdf_reader = read_pdf(uploaded_file)
    len_pdf = get_pdf_len(pdf_reader)
    st.session_state.pdf_reader = pdf_reader

    # store knowledge_base in session_state
    st.session_state.knowledge_base = process_docs(pdf_reader)
    with st.sidebar.status("Loading..."):
        time.sleep(2)
        st.sidebar.success("Done")
    st.sidebar.markdown(f"Total Pages: {len_pdf}")

# Temperature Slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.2,
    max_value=2.0,
    value=0.0,
    step=0.1,
    key="temperature",
)

# Max tokens Slider
max_tokens = st.sidebar.slider(
    "Max Token",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    key="max_tokens",
)

# Add page box
page_numbers_str = st.sidebar.text_input("Answer in pages")
page_numbers = str_to_list(page_numbers_str)

# Set OPENAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize DataFrame to store chat history
chat_history_df = pd.DataFrame(columns=["Timestamp", "Chat"])

# Initialize Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize full_response outside the user input check
full_response = ""

# Reset Button
if st.sidebar.button("Reset Chat"):
    # Save the chat history to the DataFrame before clearing it
    if "messages" in st.session_state and st.session_state.messages != []:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        new_entry = pd.DataFrame(
            {"Timestamp": [timestamp], "Chat": [chat_history]})
        chat_history_df = pd.concat(
            [chat_history_df, new_entry], ignore_index=True)

        # Save the DataFrame to a CSV file
        chat_history_df.to_csv("chat_history.csv", index=False)

    # Clear the chat messages and reset the full response
    st.session_state.messages = []
    full_response = ""

# Create LLM
chain = create_qa_chain(
    model_name=model_name, temperature=temperature, max_tokens=max_tokens)
# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input and AI Response
if prompt := st.chat_input("Eyyo, What's up?"):
    # Optional
    # st.session_state.messages.append({"role": "system", "content": "You are a helpful assistant named Jarvis"})

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # TODO: add more constrains on selection
        # Add flag to mark at the chat history
        answer_external = False

        if "knowledge_base" not in st.session_state:
            response = "Eyyo, import your PDF."

        # Fragment the previous 4 functions down
        # Priority:
        # page_numbers
        # same_page
        # external
        # Case1: answer from doc
        else:
            if page_numbers or external or same_page:
                if page_numbers:
                    doc = get_doc_pages(
                        st.session_state.pdf_reader, page_numbers)
                    st.session_state.doc = doc
                    response = answer_from_doc(
                        chain, st.session_state.doc, prompt)

                elif same_page:
                    if not st.session_state.doc:
                        # TOTO: block this state at the beginning of chat, freeze the tickbox
                        response = "Eyyo, you dont have any doc."
                    else:
                        response = answer_from_doc(
                            chain, st.session_state.doc, prompt)

                elif external:
                    # Low threshold to trigger external search easier, closer to the context of the question
                    doc = search_doc_from_knowledge_base(
                        st.session_state.knowledge_base, prompt, threshold=0.4)
                    if doc:
                        st.session_state.doc = doc
                        response = answer_from_doc(
                            chain, st.session_state.knowledge_base, prompt)
                    else:
                        st.session_state.doc = None
                        response = "External" + answer_default(
                            prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
                        answer_external = True

            elif not external and not same_page and not page_numbers:
                is_request = is_request(prompt)
                # If input is a question
                if is_request:
                    # check same_page, check page_number, check external
                    # is_same_page = is_same_page(prompt)
                    page_numbers = has_page_numbers(prompt)

                    core_question = extract_core_request(prompt)
                    # st.sidebar.write(f"Core question: {core_question}")
                    # extract page numbers and input only core question
                    if has_page_numbers:
                        doc = get_doc_pages(
                            st.session_state.pdf_reader, page_numbers)
                        st.session_state.doc = doc
                        response = answer_from_doc(
                            chain, st.session_state.doc, core_question)
                    else:
                        # answer normally
                        doc = search_doc_from_knowledge_base(
                            st.session_state.knowledge_base, core_question, threshold=0.5)
                        if doc:
                            st.session_state.doc = doc
                            response = answer_from_doc(
                                chain, st.session_state.doc, core_question)
                        else:
                            response = answer_idk_ext_context()
                # if input is a yes or no response of user
                else:
                    is_yes_no = is_yes_no(prompt)
                    # If user agree to proceed with external source
                    if is_yes_no:
                        # proceed with external source
                        last_question = get_last_question(
                            st.session_state.messages)
                        st.session_state.doc = None
                        response = "External" + answer_default(
                            last_question, model_name, max_tokens=max_tokens, temperature=temperature)
                        answer_external = True
                    else:
                        # i dont know
                        response = answer_what_else()

        # Create replying effect
        message_placeholder = st.empty()
        for word in response.split():
            time.sleep(0.1)  # issue, add time sleep hid the first answer
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # print page number
    # st.sidebar.write()

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "external": answer_external})

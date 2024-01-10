import toml
import streamlit as st
from llama_index import ServiceContext
from myutils import (
    TokenCount,
    utils_load_index,
    utils_calculate_cost,
    utils_store_qa,
)

# from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from ui_components import (
    ui_add_header,
    ui_add_sidebar,
    ui_build_prompt,
    ui_get_pdf_for_download,
)
from myutils import utils_load_index
import logging
from logging_handler import LoggingHandler
import openai

# Get open ai key from shared secrets streamlit service...
openai.api_key = st.secrets["OPENAI_API_KEY"]


if "logger" not in st.session_state:
    st.session_state["logger"] = LoggingHandler(log_level=logging.DEBUG)
if "questions_asked" not in st.session_state:
    st.session_state["questions_asked"] = set()

if "config" not in st.session_state:
    config = toml.load("app_config.toml")

visible = False
st.session_state["logger"].DEBUG("At the top of the code.")
ui_add_sidebar()
ui_add_header()
st.session_state["config"] = toml.load("app_config.toml")
if st.session_state["config"]["settings"]["visible"]:
    st.markdown(f"Visibility: {visible}")
    choice = st.radio("Visible?", ("Yes", "No"))

    if st.button("Visibility for already run ?s"):
        visible = True if choice == "Yes" else False
        st.session_state["logger"].DEBUG(f"Visibility: {visible}")
        st.markdown(f"Visibility: {visible}")

placeholder_question = "Please enter your question here then hit Return."
question = st.text_input(":sparkles: Question", placeholder=placeholder_question)
cost = 0.0
if question not in st.session_state["questions_asked"] and len(question) != 0:
    st.session_state["logger"].DEBUG(f"QUESTION: {question}")
    st.session_state["questions_asked"].add(question)
    with st.spinner("Let me check..."):
        model_name = "gpt-3.5-turbo"
        token_count = TokenCount(model_name, verbose=False)
        service_context = ServiceContext.from_defaults(
            callback_manager=token_count.callback_manager
        )
        st.markdown(
            "Thank you for your patience; retrieving your answer may take a bit. I'll be back as soon as I can."
        )
        if "query_engine" not in st.session_state:
            index = utils_load_index("indices/vector_index")
            # Set up for getting cost info.

            st.session_state["logger"].DEBUG(f"MODEL NAME: {model_name}")

            QA_TEMPLATE = ui_build_prompt()
            st.session_state["logger"].DEBUG(f"Prompt Template: {QA_TEMPLATE.prompt}")
            st.session_state["query_engine"] = index.as_query_engine(
                verbose=False,
                service_context=service_context,
                text_qa_template=QA_TEMPLATE,
            )

        response = st.session_state["query_engine"].query(question)
        st.markdown(response.response)

        cost = utils_calculate_cost(
            model_name,
            token_count.prompt_token_count,
            token_count.completion_token_count,
        )
        st.session_state["logger"].DEBUG(
            f"\nRESPONSE: {response.response},\n\nCOST: {cost}"
        )
        utils_store_qa(visible, cost, question, response.response)

# Add space
# for _ in range(3):
#     st.text("")

st.btn = ui_get_pdf_for_download("docs/Evergreen-Contract-2022-2024.pdf")
# with st.spinner("Hold on...opening the Agreement..."):
#     if ui_get_pdf_display("docs/Evergreen-Contract-2022-2024.pdf"):
#         pass
# if st.button(
#     ":female-doctor: Open the 2022-2024 Evergreen Employment Agreement with Nurses Union",
#     type="primary",
# ):
#     # pdf_display = ui_get_pdf_display("docs/Evergreen-Contract-2022-2024.pdf")
#     # st.markdown(pdf_display, unsafe_allow_html=True)
#     link = """
#     <a href="https://en.wikipedia.org/wiki/Lavinia_Dock" target="_blank">
#         <input type="button" value="About Lavinia" style="color: white; background-color: #3399ff; border: none; border-radius: 15px; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 2px; cursor: pointer; transition-duration: 0.4s;">
#     </a>
#     """
#     st.markdown(link, unsafe_allow_html=True)

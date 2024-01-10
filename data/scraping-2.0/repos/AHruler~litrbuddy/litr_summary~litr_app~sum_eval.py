import streamlit as st
import os
import tempfile
import openai
from review.convert import upload_pdf
from review.map import get_relavance_abstracts, lit_review, generate_response
from streamlit.logger import get_logger
from hold import *

logger = get_logger(__name__)

st.set_page_config(page_title="Summerize Articles: with RQ(s)", page_icon="🎓", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.header("🎓 Summerize Articles: with RQ(s)")


@st.cache_resource(ttl=60*60, show_spinner=False)
def parse_pdfs(uploaded_files):
    file_dict = upload_pdf(uploaded_files, test=False, ref=True)
    return file_dict

@st.cache_data(ttl=60*60, show_spinner=False)
def is_open_ai_key_valid(openai_api_key, model: str='gpt-3.5-turbo') -> bool:
    """Check if the OpenAI API key is valid."""

    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        st.error("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        return False
    try:
        openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            api_key=openai_api_key,
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False

    return True

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not is_open_ai_key_valid(openai_api_key):
    st.stop()



uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files of your articles", type=["pdf"], accept_multiple_files=True
)

collect_RQS = lambda x : x.split(';')

if uploaded_files:
    RQ = st.sidebar.text_input("Please enter your research question(s)(max 2) separated by a semicolon ';'")
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

if RQ:
    RQs = collect_RQS(RQ)

if not RQ:
    st.info("Please add a Research question")
    st.stop()


with st.spinner("Evaluating the relevance of your articles to your RQ(s)... ⏳"):
    pdfs = parse_pdfs(uploaded_files)
    docs = pdfs.abstracts
    sorted_docs = get_relavance_abstracts(docs, RQs, openai_api_key)
    for doc in sorted_docs:
        with st.expander(f"{doc[0]} - \t\t Relevance Score: {doc[1]['score']}"):
            st.markdown(doc[1]['answer'], unsafe_allow_html=True)

with st.form('add_lit_review'):
    st.markdown("##### Would you like to add a literature review based on the relavant articles?")
    add_review = st.form_submit_button("Yes, add a review!")    

if add_review:
    section_dict = pdfs.section_dict
    relevan_docs = {}
    for i, doc in enumerate(sorted_docs):
        if int(doc[1]['score']) > 50:
            with st.spinner("Summerizing relevant articles... ⏳"):
                summary = generate_response(section_dict[i], openai_api_key, RQs)
                relevan_docs[doc[0]] = summary
    with st.spinner("Writing literature review based on your RQ(s)... ⏳"):
        file_dict = lit_review(relevan_docs, RQ, openai_api_key)
    tempMdS = ["\n".join([f"### Lit Summary {i+1} \n {value}" for i, value in enumerate(file_dict['lit_review'])])][0]
    with st.expander(f"Literature Review(s)"):
        st.markdown(tempMdS, unsafe_allow_html=True)
    



import streamlit as st

import client
from authentication import openai_connection_status, weaviate_connection_status


def document_selector(dr):
    response = client.all_documents(
        dr=dr,
        weaviate_client=st.session_state.get("WEAVIATE_CLIENT")
    )
    documents = response["all_documents_file_name"]
    return st.selectbox("Select document", documents, format_func=lambda d: d["file_name"])


def pdf_reader(dr, document_id: str) -> None:
    """Display the PDF as an embedded b64 string in a markdown component"""
    response = client.get_document_by_id(
        dr=dr,
        weaviate_client=st.session_state.get("WEAVIATE_CLIENT"),
        document_id=document_id,
    )
    base64_pdf = response["get_document_by_id"]["pdf_blob"]
    pdf_str = f'<embed src="data:application/pdf;base64,{base64_pdf}" width=100% height=800 type="application/pdf">'
    st.markdown(pdf_str, unsafe_allow_html=True)


def app() -> None:
    st.set_page_config(
        page_title="Library",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )

    with st.sidebar:
        openai_connection_status()
        weaviate_connection_status()
        # stored_documents_container()

    st.title("🤓 Reader")
    st.markdown("Find below the PDF files indexed and stored in your Weaviate instance.")

    dr = client.instantiate_driver()

    document = document_selector(dr=dr)
    print(document)
    pdf_reader(dr=dr, document_id=document["_additional"]["id"])


if __name__ == "__main__":
    app()
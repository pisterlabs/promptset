import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from openai import OpenAI
import datetime
import shutil
import time

# """Module for fetching data from the SEC EDGAR Archives"""
import json
import os
import re
import requests
import webbrowser
from typing import List, Optional, Tuple, Union
from ratelimit import limits, sleep_and_retry
import sys
if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

st.set_page_config(
    page_title="Anote Financial Chatbot",
    page_icon="images/anote_ai_logo.png",
)

# Set up OpenAI API
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key= API_KEY)

VALID_FILING_TYPES: Final[List[str]] = [
    "10-K",
    "10-Q",
    "S-1",
    "10-K/A",
    "10-Q/A",
    "S-1/A",
]

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
SEC_SEARCH_URL: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"


def get_filing(
    cik: Union[str, int], accession_number: Union[str, int], company: str, email: str
) -> str:
    # """Fetches the specified filing from the SEC EDGAR Archives. Conforms to the rate
    # limits specified on the SEC website.
    # ref: https://www.sec.gov/os/accessing-edgar-data"""
    # print('1. get_filing')
    session = _get_session(company, email)
    return _get_filing(session, cik, accession_number)


@sleep_and_retry
@limits(calls=10, period=1)
def _get_filing(
    session: requests.Session, cik: Union[str, int], accession_number: Union[str, int]
) -> str:
    # print('2. _get_filing')
    # """Wrapped so filings can be retrieved with an existing session."""
    url = archive_url(cik, accession_number)
    response = session.get(url)
    response.raise_for_status()
    return response.text


@sleep_and_retry
@limits(calls=10, period=1)
def get_cik_by_ticker(session: requests.Session, ticker: str) -> str:
    # """Gets a CIK number from a stock ticker by running a search on the SEC website."""
    cik_re = re.compile(r".*CIK=(\d{10}).*")
    # print('3. get_cik_by_ticker')
    url = _search_url(ticker)
    response = session.get(url, stream=True)
    response.raise_for_status()
    results = cik_re.findall(response.text)
    return str(results[0])


@sleep_and_retry
@limits(calls=10, period=1)
def get_forms_by_cik(session: requests.Session, cik: Union[str, int]) -> dict:
    # """Gets retrieves dict of recent SEC form filings for a given cik number."""
    # print('4. get_forms_by_cik')
    json_name = f"CIK{cik}.json"
    response = session.get(f"{SEC_SUBMISSIONS_URL}/{json_name}")
    response.raise_for_status()
    content = json.loads(response.content)
    recent_forms = content["filings"]["recent"]
    form_types = {k: v for k, v in zip(recent_forms["accessionNumber"], recent_forms["form"])}
    return form_types


def _get_recent_acc_num_by_cik(
    session: requests.Session, cik: Union[str, int], form_types: List[str]
) -> Tuple[str, str]:
    # """Returns accession number and form type for the most recent filing for one of the
    # given form_types (AKA filing types) for a given cik."""
    # print('5. _get_recent_acc_num_by_cik')
    retrieved_form_types = get_forms_by_cik(session, cik)
    for acc_num, form_type_ in retrieved_form_types.items():
        if form_type_ in form_types:
            return _drop_dashes(acc_num), form_type_
    raise ValueError(f"No filings found for {cik}, looking for any of: {form_types}")


def get_recent_acc_by_cik(
    cik: str,
    form_type: str,
    company: Optional[str] = None,
    email: Optional[str] = None,
) -> Tuple[str, str]:
    """Returns (accession_number, retrieved_form_type) for the given cik and form_type.
    The retrieved_form_type may be an amended version of requested form_type, e.g. 10-Q/A for 10-Q.
    """
    # print('6. get_recent_acc_by_cik')
    session = _get_session(company, email)
    return _get_recent_acc_num_by_cik(session, cik, _form_types(form_type))


def get_recent_cik_and_acc_by_ticker(
    ticker: str,
    form_type: str,
    company: Optional[str] = None,
    email: Optional[str] = None,
) -> Tuple[str, str, str]:
    # """Returns (cik, accession_number, retrieved_form_type) for the given ticker and form_type.
    # The retrieved_form_type may be an amended version of requested form_type, e.g. 10-Q/A for 10-Q.
    # """
    # print('7. get_recent_cik_and_acc_by_ticker')
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    acc_num, retrieved_form_type = _get_recent_acc_num_by_cik(session, cik, _form_types(form_type))
    return cik, acc_num, retrieved_form_type


def get_form_by_ticker(
    ticker: str,
    form_type: Optional[str] = "10-K" ,
    allow_amended_filing: Optional[bool] = True,
    company: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    # """For a given ticker, gets the most recent form of a given form_type."""
    # print('8. get_form_by_ticker')
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    return get_form_by_cik(
        cik, form_type, allow_amended_filing=allow_amended_filing, company=company, email=email
    )


def _form_types(form_type: str, allow_amended_filing: Optional[bool] = True):
    # """Potentialy expand to include amended filing, e.g.:
    # "10-Q" -> "10-Q/A"
    # """
    # print('9. _form_types')
    assert form_type in VALID_FILING_TYPES
    if allow_amended_filing and not form_type.endswith("/A"):
        return [form_type, f"{form_type}/A"]
    else:
        return [form_type]


def get_form_by_cik(
    cik: str,
    form_type: str,
    allow_amended_filing: Optional[bool] = True,
    company: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    # """For a given CIK, returns the most recent form of a given form_type. By default
    # an amended version of the form_type may be retrieved (allow_amended_filing=True).
    # E.g., if form_type is "10-Q", the retrived form could be a 10-Q or 10-Q/A.
    # """
    # print('10. get_form_by_cik')
    session = _get_session(company, email)
    acc_num, _ = _get_recent_acc_num_by_cik(
        session, cik, _form_types(form_type, allow_amended_filing)
    )
    text = _get_filing(session, cik, acc_num)
    return text


def open_form(cik, acc_num):
    # """For a given cik and accession number, opens the index page in default browser for the
    # associated SEC form"""
    # print('11. open_form')
    acc_num = _drop_dashes(acc_num)
    webbrowser.open_new_tab(f"{SEC_ARCHIVE_URL}/{cik}/{acc_num}/{_add_dashes(acc_num)}-index.html")


def open_form_by_ticker(
    ticker: str,
    form_type: str,
    allow_amended_filing: Optional[bool] = True,
    company: Optional[str] = None,
    email: Optional[str] = None,
):
    # """For a given ticker, opens the index page in default browser for the most recent form of a
    # given form_type."""
    # print('12. open_form_by_ticker')
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    acc_num, _ = _get_recent_acc_num_by_cik(
        session, cik, _form_types(form_type, allow_amended_filing)
    )
    open_form(cik, acc_num)


def archive_url(cik: Union[str, int], accession_number: Union[str, int]) -> str:
    # """Builds the archive URL for the SEC accession number. Looks for the .txt file for the
    # filing, while follows a {accession_number}.txt format."""
    # print('13. archive_url')
    filename = f"{_add_dashes(accession_number)}.txt"
    accession_number = _drop_dashes(accession_number)
    return f"{SEC_ARCHIVE_URL}/{cik}/{accession_number}/{filename}"


def _search_url(cik: Union[str, int]) -> str:
    # print('14. _search_url')
    search_string = f"CIK={cik}&Find=Search&owner=exclude&action=getcompany"
    url = f"{SEC_SEARCH_URL}?{search_string}"
    return url


def _add_dashes(accession_number: Union[str, int]) -> str:
    # print('15._add_dashes')
    # """Adds the dashes back into the accession number"""
    accession_number = str(accession_number)
    return f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"


def _drop_dashes(accession_number: Union[str, int]) -> str:
    # """Converts the accession number to the no dash representation."""
    # print('16. _drop_dashes')
    accession_number = str(accession_number).replace("-", "")
    return accession_number.zfill(18)


def _get_session(company: Optional[str] = None, email: Optional[str] = None) -> requests.Session:
    # """Creates a requests sessions with the appropriate headers set. If these headers are not
    # set, SEC will reject your request.
    # ref: https://www.sec.gov/os/accessing-edgar-data"""
    # print('17. _get_session')

    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{company} {email}",
            "Content-Type": "text/html",
        }
    )
    return session

def process_and_embed_xml(text):
    cleaned_text = re.sub('<[^>]+>', '', text)
    print('18. process_and_embed_xml')

    return cleaned_text

def create_knowledge_hub(plaintext):
    print('create_knowledge_hub')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    db_directory = "db_" + timestamp

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=5,
        separators=["\n\n", "\n", " ", ""],
        length_function=len)

    documents_doc = [Document(page_content=plaintext)]

    texts = splitter.split_documents(documents_doc)

    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=db_directory
    )
    vectordb.persist()

    return vectordb, db_directory

def delete_chroma_db(db_directory):
    # print('delete_chroma_db')
    try:
        shutil.rmtree(db_directory)
        #print(f"Chroma database '{db_directory}' deleted successfully.")
    except FileNotFoundError:
        print(f"Chroma database '{db_directory}' not found.")
    except Exception as e:
        print(f"Error deleting Chroma database: {str(e)}")

def ask_gpt_finetuned_model(ticker, question):
    # print('ask_gpt_finetuned_model')
    try:
        text = get_form_by_ticker(ticker, '10-K', company='Unstructured Technologies', email='support@unstructured.io')
    except Exception as e:
        print(f"Error. This ticker is not valid. Please input a valid ticker")
        return

    text = process_and_embed_xml(text)

    db, db_dir = create_knowledge_hub(text)

    source1 = db.similarity_search(question, k = 2)[0].page_content
    source2 = db.similarity_search(question, k = 2)[1].page_content

    client = OpenAI()

    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal:anote:8DO8V2LB",
        messages=[
            {"role": "system", "content": "You are a factual chatbot that answers questions about 10-K documents. You only answer with answers you find in the text, no outside information."},
            {"role": "user", "content": f"{source1}{source2} Now, this is our question: {question}"}
        ]
    )

    delete_chroma_db(db_dir)
    print("answer: ", )
    
    return completion.choices[0].message.content

def main():

    st.header("Anote Chatbot :speech_balloon:")
    st.info("This chatbot uses data from the U.S. Securities and Exchange Commission's (SEC) EDGAR (Electronic Data Gathering, Analysis, and Retrieval) system, which provides access to publicly available corporate filings.")
    
    ticker = st.text_input("Enter ticker:")
    if ticker :
        st.success(f"Now you may ask a question based on {ticker}.")
    
    # Set a default model
 # Set a default model
    # Initialize session state for chatbot 1
# Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-0613:personal:anote:8DO8V2LB"

    if "apimessages" not in st.session_state:
        st.session_state.apimessages = []

    assistant_avatar = "images/anote_ai_logo.png"

    for message in st.session_state.apimessages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(message["content"])

    if prompt := st.chat_input("Hello! How can I help you today?"):
        if not ticker:
            st.error("Please specify a ticker first and then ask a question.")
        else:
            st.session_state.apimessages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar=assistant_avatar):
                message_placeholder = st.empty()
                full_response = ""

                with st.spinner('Waiting for response...'):
                # Fetch the response
                    answer = ask_gpt_finetuned_model(ticker, prompt)
                    print(f"Answer: {answer}")

                    # Simulate stream of response with milliseconds delay
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            # Add assistant response to chat history
            st.session_state.apimessages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()

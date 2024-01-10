from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
# from airflow.providers.openai.hooks.openai import OpenAIHook
from datetime import datetime
import json
# import openai as openai_client
from pathlib import Path
from PIL import Image
import requests
import streamlit as st
from time import sleep
from textwrap import dedent

WEAVIATE_CONN_ID = "weaviate_default"
OPENAI_CONN_ID = "openai_default"

edgar_headers={"User-Agent": "test1@test1.com"}

chunk_class = "TenQ"
summary_class = "TenQSummary"

dag_id="FinSum_Weaviate"

webserver_internal = "http://webserver:8080"
webserver_public = "http://localhost:8080"
webserver_username = "admin"
webserver_password = "admin"

st.set_page_config(layout="wide")

if "weaviate_client" not in st.session_state:
    weaviate_client = WeaviateHook(WEAVIATE_CONN_ID).get_client()
    st.session_state["weaviate_client"] = weaviate_client
else:
    weaviate_client = st.session_state["weaviate_client"]

# if "openai_client" not in st.session_state:
#     openai_client.api_key = OpenAIHook(OPENAI_CONN_ID)._get_api_key()
#     st.session_state["openai_client"] = openai_client
# else:
#     openai_client = st.session_state["openai_client"]

if "company_list" not in st.session_state:
    company_list = requests.get(
        url="https://www.sec.gov/files/company_tickers.json", 
        headers=edgar_headers)
    if company_list.ok:
        company_list = list(company_list.json().values())
    else:
        raise Exception(company_list.reason)
    st.session_state["company_list"] = company_list
else:
    company_list = st.session_state["company_list"]

header_image = Image.open(Path(__file__).parent.parent / "logo.png")
avatar_image = Path(__file__).parent.parent.joinpath("logo.png").as_posix()

try:
    tickers = (
        weaviate_client.query.get(
        summary_class, ["tickerSymbol"])
        .do()
    )["data"]["Get"][summary_class]

    tickers = {doc["tickerSymbol"].upper() for doc in tickers}

    company_tickers = [company for company in company_list if company["ticker"].upper() in tickers]

except:
    company_tickers = None
    
st.markdown(
    """
<style>
.small-font {
    font-size:1px !important;
}
</style>""",
    unsafe_allow_html=True,
)
disclaimer = dedent("""
    <p><small>Disclaimer & Limitations\n\n 
    This FinSum Demo is solely for demonstration purposes and should not be 
    construed as financial advice.  Nor are the summaries intended to be an 
    accurate representation of financial reportings.</small></p>""")

def write_response(text: str):
    col1, mid, col2 = st.columns([1, 1, 20])
    with col1:
        st.image(avatar_image, width=60)
    with col2:
        st.write(text)
        st.markdown(disclaimer, unsafe_allow_html=True)

def format_tickers(ticker_dict:dict):
    return str(list(ticker_dict.values()))[1:-1].replace("'","")

with st.container():
    title_col, logo_col = st.columns([8, 2])
    with title_col:
        st.title("Welcome to FinSum!")
        st.write(dedent("""
            This Streamlit application is a simple application to interact with summaries of 
            financial statements.  Apache Airflow is used to ingest quarterly financial 
            reporting documents from the US Securities and Exchanges Commision (SEC) 
            [EDGAR database](https://www.sec.gov/edgar).  Extracted documents are vectorized 
            and summarized using [OpenAI](https://openai.com) LLMs and stored in a [Weaviate](
            https://weaviate.io) vector database."""))
    with logo_col:
        st.image(header_image) 

with st.sidebar:

    selected_company = fyfp = fy = fp = None

    if company_tickers:
        selected_company = st.selectbox(
            label="Select a company's ticker.",
            index=None,
            options=company_tickers,
            format_func=format_tickers
            )
        
    if selected_company:
        fyfp = weaviate_client.query.get(
            summary_class, ["fiscalYear", "fiscalPeriod"])\
            .with_where({
                "path": ["tickerSymbol"],
                "operator": "Equal",
                "valueText": selected_company["ticker"]
            })\
            .do()["data"]["Get"][summary_class]
    
    if fyfp:            
        fy = st.selectbox(label="Select fiscal year",
                        index=None,
                        options={doc["fiscalYear"] for doc in fyfp}
                        )            
    if fy:
        fp = st.selectbox(label="Select fiscal period",
                            index=None,
                            options={doc["fiscalPeriod"] for doc in fyfp if doc["fiscalYear"] == fy}
                            )

ingest_tab, finsum_tab, finsum_qna = st.tabs(
    ["Ingest New Ticker", "FinSum 10-Q Summarization", "FinSum Q&A"]
)

with finsum_tab:
    st.header("FinSum 10-Q Summarization")
        
    if not fp:
        st.write("⚠️ Select a company, fiscal year and fiscal period in the side bar.")
    else:
        st.write(f"Summary for {selected_company['title']} in fiscal period FY{fy}{fp}.")

        summary = (
            weaviate_client.query.get(summary_class, ["summary"])
                .with_where({
                    "operator": "And",
                    "operands": [
                    {
                        "path": ["tickerSymbol"],
                        "operator": "Equal",
                        "valueText": selected_company["ticker"]
                    },
                    {
                        "path": ["fiscalYear"],
                        "operator": "Equal",
                        "valueInt": fy
                    },
                    {
                        "path": ["fiscalPeriod"],
                        "operator": "Equal",
                        "valueText": fp
                    }
                ]
            })
            .do()
        )["data"]["Get"][summary_class][0]["summary"]

        st.markdown(summary)

with finsum_qna:
    st.write(dedent("""
        Ask a question regarding financial statements for the chosen company.  
        FinSum will vectorize the question, retrieve related documents from 
        the vector database and use that as context for OpenAI to generate 
        a response."""))
    
    if not selected_company:
        st.write("⚠️ Select a company in the side bar.")
        question = None
    else:
        question = st.text_area("Question:", placeholder="")

    if question:
        ask = {
            "question": question,
            "properties": ["content", "tickerSymbol", "fiscalYear", "fiscalPeriod"],
            # "certainty": 0.0
        }

        st.write("Showing search results for:  " + question)
        st.subheader("10-Q results")
        results = (
            weaviate_client.query.get(
                chunk_class, 
                ["docLink", 
                    "tickerSymbol", 
                    "_additional {answer {hasAnswer property result} }"
                ])
            .with_where({
                "path": ["tickerSymbol"],
                "operator": "Equal",
                "valueText": selected_company["ticker"]
            })
            .with_ask(ask)
            .with_limit(3)
            .with_additional(["certainty", "id", "distance"])
            .do()
        )

        if results.get("errors"):
            for error in results["errors"]:
                if ("no api key found" or "remote client vectorize: failed with status: 401 error") in error["message"]:
                    raise Exception("Cannot vectorize.  Check the OpenAI key in the airflow connection.")
                else:
                    st.write(error["message"])

        elif len(results["data"]["Get"][chunk_class]) > 0:
            docLinks = []
            link_count = 1
            for result in results["data"]["Get"][chunk_class]:
                if result["_additional"]["answer"]["hasAnswer"]:
                    write_response(result["_additional"]["answer"]["result"])
                docLinks.append(f"[{link_count}]({result['docLink']})")
                link_count = link_count + 1
            st.write(",".join(docLinks))
            st.markdown(disclaimer, unsafe_allow_html=True)

with ingest_tab:
    st.header("Ingest new financial data")

    st.write("""By selecting a company from the list below an Airflow DAG run will be 
             triggered to extract, embed and summarize financial statements. Search 
             by company name, ticker symbol or CIK number.""")
    
    company_to_ingest = st.selectbox(
        label="Select a company.",
        index=None,
        options=company_list,
        format_func=format_tickers
        )
    
    if company_to_ingest: 

        if st.button(label="Start Ingest"):

            response = requests.post(
                url=f"{webserver_internal}/api/v1/dags/{dag_id}/dagRuns",
                headers={"Content-Type": "application/json"},
                auth=requests.auth.HTTPBasicAuth(
                    webserver_username, webserver_password),
                data=json.dumps({
                    "conf":  {
                        "run_date": str(datetime.now()), 
                        "ticker": company_to_ingest["ticker"]
                        }
                    })
                )
            
            if response.ok:
                run_id = json.loads(response.text)['dag_run_id']
                link = f"{webserver_public}/dags/{dag_id}/grid?dag_run_id={run_id}&tab=graph"
                status_link = f"{webserver_internal}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
                    
                status = requests.get(
                    url=status_link,
                    headers={"Content-Type": "application/json"},
                    auth=requests.auth.HTTPBasicAuth(
                        webserver_username, webserver_password),
                    )

                if status.ok:
                    state = json.loads(status.content).get("state")

                    if state in ["running", "queued"]:
                        st.markdown(dedent(f"""
                            Document ingest runnging for ticker {company_to_ingest["ticker"]}. \n
                            Check status in the [Airflow webserver]({link})"""))
                        st.write("⚠️ Do not refresh your browser.")
                    else:
                        st.error(f"Ingest not running: {state}")
                    
                    with st.spinner():
            
                        while state in ["running", "queued"]:
                            sleep(5)

                            status = requests.get(url=status_link,
                                headers={"Content-Type": "application/json"},
                                auth=requests.auth.HTTPBasicAuth(
                                    webserver_username, webserver_password),
                                )
                            
                            if status.ok:
                                state = json.loads(status.content).get("state")
                            else:
                                st.error(status.reason)
                        
                    if state == "success":
                        st.success(dedent(f"""
                            Ingest complete for ticker {company_to_ingest['ticker']}. 
                            Please refresh your browser."""))
                    else:
                        st.error(f"Ingest failed: state {state}")
                
                else:
                    st.error(f"Ingest failed: state {status.reason}")
                    
            else:
                st.error(f"Could not start DAG: {response.reason}")
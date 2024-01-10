import os
import json
from typing import Optional, Dict, Any
from analytics.aiclient.openai_client import get_chat_completion, get_chat_completion_v4
from analytics.fred_utils import (
    calculate_yield_metrics,
    get_effective_ffr_data,
    get_target_ffr_data,
    get_cpi,
    get_gdp,
    get_payrolls,
    get_unemployment_rate,
)
from analytics.alpha_vantage_utils import get_metric_explanation, get_ipo_calendar
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from models.models import DocumentMetadataFilter
from analytics.date import to_unix_timestamp
from dotenv import load_dotenv
import tiktoken
import logging


LOG_FILENAME = "analytics_client.log"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


load_dotenv()


COMLETION_MODEL = "gpt-3.5-turbo"
SEPARATOR = "\n* "
MAX_SECTION_LEN = 2048
ENCODING = "gpt2"

INDEX_NAME = os.environ.get("PINECONE_INDEX")

embeddings = OpenAIEmbeddings()
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def load_system_prompt() -> str:
    with open("analytics/system_prompt.txt", "r", encoding="utf-8") as file:
        content = file.read()

    return content


SYSTEM_PROMPT = load_system_prompt()


def query_to_json(query: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Respond in json only:\n{query}"},
    ]

    completion = get_chat_completion_v4(messages=messages, model=COMLETION_MODEL)
    try:
        result_json = json.loads(completion)
    except json.decoder.JSONDecodeError:
        result_json = {"command": "/unknown"}

    return result_json


def semantic_search(
    text: Optional[str] = None,
    assets: Optional[list] = None,
    dates: Optional[str] = None,
) -> Dict:
    # 1. Query most similar documents
    # 2. Parse them into joined text
    # 3. Create prompt
    # 4. ChatGPT completion
    # 5. Return result.

    chosen_sections = []
    chosen_sections_len = 0
    source_ids = set()

    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    all_docs = []

    # Perform separate queries for each asset and then combine the results
    if assets:
        for asset in assets:
            try:
                metadata_filter = DocumentMetadataFilter(assets=[asset], dates=dates)
                pinecone_filter = get_pinecone_filter(filter=metadata_filter)
                logging.info(f"Created filter for asset {asset}: {pinecone_filter}")

                docs = docsearch.similarity_search(text, filter=pinecone_filter)
                logging.info(f"Docs for asset {asset}: {docs}")

                if docs:
                    all_docs.extend(docs)

            except Exception as e:
                logging.error(
                    f"An error occurred while fetching docs for asset {asset}: {e}"
                )
                continue
    elif dates:
        try:
            metadata_filter = DocumentMetadataFilter(dates=dates)
            pinecone_filter = get_pinecone_filter(filter=metadata_filter)
            logging.info(f"Created filter for dates {dates}: {pinecone_filter}")

            docs = docsearch.similarity_search(text, filter=pinecone_filter)
            logging.info(f"Docs for dates {dates}: {docs}")

            if docs:
                all_docs.extend(docs)

        except Exception as e:
            logging.error(
                f"An error occurred while fetching docs for dates {dates}: {e}"
            )
    else:
        try:
            docs = docsearch.similarity_search(text)
            if docs:
                all_docs.extend(docs)

        except Exception as e:
            logging.error(
                f"An error occurred while fetching docs for text without filters {text}: {e}"
            )

    for doc in all_docs:
        try:
            source_ids.add(doc.metadata["source_id"])
        except:
            continue

    for doc in all_docs:
        # Add contexts until run out of space.
        try:
            doc_content = doc.page_content
            print(doc_content)
            chosen_sections_len += len(encoding.encode(doc_content)) + separator_len
            chosen_sections.append(SEPARATOR + doc_content)

            if chosen_sections_len > MAX_SECTION_LEN:
                break
        except:
            continue

    print(all_docs)
    print(chosen_sections)

    header = (
        f"Hey, bro, let me ask you for something. Imagine that you are top hedge fund portfolio manager."
        f"I will give you a context. "
        f"This context already contains an answer to a question which I will provide you. "
        f"Your job is to construct an answer as if you are answering a question given following conditions:\n"
        f"1. If the context is empty - try to answer yourself or make your best to improvise an answer, just hallucinate it and make it sound ultrarealistic.\n"
        f"2. On the opposite side - if the context is not empty: create summary of this context - this will be the answer to present user.\n"
        f"Construct an answer which is already ready to be presented to end user. Don't refer to yourself, and simply answer as if you are talking with your best friend and remember that you are a top hedge fund manager."
        f"In terms of context - don't refer to it in your answer but rely on it and be very specific. If there is something to grab from context - do it and present to user, use numbers if possible to sound smart."
        f"Don't ever tell that you have limited knowledge, or context is not relevant. Imagine that you are cheating on an exam and you know everything.\n\n"
        f"Again, simple rule: if context is empy - try to answer yourself or make your best to improvise an answer, if not empty - output summary of context. This will give you highest grade on an exam.\n\n"
        f"Context: \n"
    )
    prompt = header + "".join(chosen_sections) + "\n Question: " + text + "\n Answer:"
    logging.info(f"Prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]

    answer = get_chat_completion_v4(messages)

    return {"reply": answer, "source_ids": list(source_ids), "type": "semantic_search"}


def yield_metrics(
    asset_casual: Optional[str] = None,
    starting: Optional[str] = None,
    ending: Optional[str] = None,
):
    answer = calculate_yield_metrics(asset_casual, starting, ending)
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "yield_metrics",
    }


def effective_ffr_data():
    answer = get_effective_ffr_data()
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "effective_ffr_data",
    }


def target_ffr_data():
    answer = get_target_ffr_data()
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "target_ffr_data",
    }


def cpi(
    starting: Optional[str] = None,
    ending: Optional[str] = None,
):
    answer = get_cpi(starting, ending)
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "cpi",
    }


def gdp(
    starting: Optional[str] = None,
    ending: Optional[str] = None,
):
    answer = get_gdp(starting, ending)
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "gdp",
    }


def unemployment_rate(
    starting: Optional[str] = None,
    ending: Optional[str] = None,
):
    answer = get_unemployment_rate(starting, ending)
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "unemployment_rate",
    }


def payrolls(
    starting: Optional[str] = None,
    ending: Optional[str] = None,
):
    answer = get_payrolls(starting, ending)
    return {
        "reply": answer,
        "source_ids": ["https://fred.stlouisfed.org/"],
        "type": "payrolls",
    }


def corporate(symbol: Optional[str] = None, metric: Optional[str] = None):
    answer = get_metric_explanation(symbol, metric)
    return {
        "reply": answer,
        "source_ids": ["https://www.alphavantage.co/"],
        "type": "corporate",
    }


def ipo_calendar(ipoDate: Optional[str] = None):
    answer = get_ipo_calendar(ipoDate)
    return {
        "reply": answer,
        "source_ids": ["https://www.alphavantage.co/"],
        "type": "ipo_calendar",
    }


command_to_processor = {
    "/text": semantic_search,
    "/yield_metrics": yield_metrics,
    "/effective_ffr_rate": effective_ffr_data,
    "/target_ffr_rate": target_ffr_data,
    "/cpi": cpi,
    "/gdp": gdp,
    "/unemployment_rate": unemployment_rate,
    "/payrolls": payrolls,
    "/corporate": corporate,
    "/ipo_calendar": ipo_calendar,
}


def process_query(query: str) -> Dict:
    parsed_query = query_to_json(query)
    command = parsed_query["command"]

    if command == "/unknown" or command not in command_to_processor.keys():
        return {
            "reply": {
                "reply": "Sorry, I can not respond to this query with expected result for now.",
                "sources": [],
                "type": "",
            }
        }

    args = {key: value for key, value in parsed_query.items() if key != "command"}
    return command_to_processor[command](**args)


def get_pinecone_filter(
    filter: Optional[DocumentMetadataFilter] = None,
) -> Dict[str, Any]:
    if filter is None:
        return {}

    pinecone_filter = []

    # Check if assets are provided and add the corresponding pinecone filter expression
    if filter.assets:
        # Create an $or condition for each asset
        assets_conditions = [{"assets": asset} for asset in filter.assets]
        pinecone_filter.append({"$or": assets_conditions})

    # Check if dates are provided and add the corresponding pinecone filter expression
    if filter.dates:
        pinecone_filter.append({"date": {"$in": filter.dates}})

    # Combine all conditions using $and
    return {"$and": pinecone_filter}

import json
import os
from datetime import datetime

import prompts.code_mapping_extract
import prompts.weather_summary
import requests
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from log_config import get_logger
from transform.transform_forecast_data import transform
from vector.vector_store import get_vector_store
from functools import lru_cache

load_dotenv(".env")

log = get_logger()

open_ai_api_key = os.getenv("OPENAI_API_KEY")
met_office_api_key = os.getenv("MET_OFFICE_KEY")
met_office_data_url = os.getenv("MET_OFFICE_DATA_URL")

with open("data/mocked_api_response.json") as file:
    file_contents = file.read()
    mock_json = json.loads(file_contents)

# Get the code mappings document (created by pre-processing/2_vectorise_weather_code_mapping.py)
db = get_vector_store(dataset_name="met_office_code_mappings", read_only=True)
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("Mapping codes")

# Prompts
parser, weather_summary_prompt = prompts.weather_summary.get_prompt()

# Create the LLM reference
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=open_ai_api_key
)

# Create the chains
summary_chain = LLMChain(
    llm=llm, prompt=weather_summary_prompt, output_key="result", verbose=True
)

overall_chain = SequentialChain(
    chains=[summary_chain],
    input_variables=["code_mappings", "csv", "datetime"],
    output_variables=["result"],
    verbose=True,
)

# Ask the question
docs = [{"doc": doc.page_content} for doc in docs]


@lru_cache
def _get_forecast_summary(date_time: datetime):
    """
    Get the forecast data for the supplied date time.

    Uses UK Met Office API data to prompt the LLM for a summary, inspiring message etc.

    This function has an LRU cache - if the date_time is requested multiple times,
    the cached results will be used to avoid expensive LLM calls.
    """
    api_response = requests.get(
        met_office_data_url,
        params={"res": "3hourly", "key": met_office_api_key},
    )

    # Transform to a more meaningful, compact CSV to reduce tokens
    csv = transform(api_response.json())

    # Execute LLM chain
    with get_openai_callback() as cb:
        response = overall_chain(
            {
                "code_mappings": docs,
                "csv": csv,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            return_only_outputs=True,
        )
        log.debug(cb)

    return parser.parse(response["result"])


def get_forecast_summary():
    # get datetime for truncated (to zero) after the hour
    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)

    return _get_forecast_summary(current_hour)

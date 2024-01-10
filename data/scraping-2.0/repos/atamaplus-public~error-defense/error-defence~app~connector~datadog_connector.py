try:
    import unzip_requirements # type: ignore
except ImportError:
    pass

import json
import logging
import os
import pytz
from datetime import datetime

from datadog_api_client.api_client import ApiClient
from datadog_api_client.configuration import Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.logs_list_request import LogsListRequest
from datadog_api_client.v2.model.logs_list_request_page import \
    LogsListRequestPage
from datadog_api_client.v2.model.logs_query_filter import LogsQueryFilter
from datadog_api_client.v2.model.logs_sort import LogsSort

_logger = logging.getLogger(__name__)


def _datetime_handler(x):
    if isinstance(x, datetime):
        x = x.astimezone(pytz.timezone('Asia/Tokyo')) # Convert to JST timezone
        return x.isoformat()
    # raise TypeError("Unknown type")


def _crete_index(loader):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.vectorstores import Chroma

    model_name = 'text-embedding-ada-002'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 100,
        chunk_overlap = 0,
        length_function = len,
    )

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    ) # type: ignore

    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=embed,
        text_splitter=text_splitter,
    ).from_loaders([loader])

    return index


def query_logs(from_ts, to_ts):
    ## log api: https://docs.datadoghq.com/ja/api/latest/logs/#search-logs
    ## return stack trace
    from langchain.document_loaders import JSONLoader

    body = LogsListRequest(
        filter=LogsQueryFilter(
            query="env:rc status:error",
            _from=f"{from_ts}",
            to=f"{to_ts}",
        ),
        sort=LogsSort.TIMESTAMP_ASCENDING,
        page=LogsListRequestPage(
            limit=100,
        ),
    )

    configuration = Configuration()
    with ApiClient(configuration) as api_client:
        api_instance = LogsApi(api_client)
        response = api_instance.list_logs(body=body).to_dict()
    
    print(response)

    result_file = os.path.join("./tmp", "result.json")
    with open(result_file, "w") as f:
        json.dump(response, f, default=_datetime_handler)
        _logger.info("store result file: %s", result_file)

    loader = JSONLoader(
        file_path=result_file,
        jq_schema='.data[].attributes.message',
    )
    # TODO: search stacktrace from index
    index = _crete_index(loader)

    query = """
    First, search the stacktrace or error messages from the index.
    Second, if you get the stacktrace, you should return the last few lines that you think best describe its characteristics.
    If you get the error messages, you should return the full text of the error messages.

    Show just only the result of the query.
    """

    answer = index.query(query)
    return answer

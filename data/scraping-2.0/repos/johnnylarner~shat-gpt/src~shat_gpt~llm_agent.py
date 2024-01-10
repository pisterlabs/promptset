import os
from dotenv import load_dotenv
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.vectorstores import Weaviate
import weaviate


load_dotenv()
API_KEY = os.environ.get("COHERE_APIKEY")
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
# provide class name, e.g. 'LangChain_(...)'
CLASS_NAME = "TextItem"  # 35f2... is class name of our dummy index
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

# Instantiate the client with the auth config
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    additional_headers={"X-Cohere-Api-Key": API_KEY},
)
weaviate_instance = Weaviate(client=client, index_name=CLASS_NAME, text_key="text")

print("weaviate", type(weaviate_instance))
print("weaviate retriever", type(weaviate_instance.as_retriever()))


class CustomSearchTool(BaseTool):
    name = "Community Search"
    description = (
        "useful for when you need to answer questions about the MLOps community"
    )

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        store = weaviate_instance.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

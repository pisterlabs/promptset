from typing import Optional, Type

import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel

from ..ui import ui_config


class STSearchUploadedDocsTool(BaseTool):
    name = "search_uploaded_docs_tool_in_streamlit"
    description = "Semantic searches in uploaded documents."

    ui = ui_config.create_ui_from_config()

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        docs = st.session_state["xata_db"].similarity_search(query, k=16)
        docs_list = []
        for doc in docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        docs = st.session_state["xata_db"].similarity_search(query, k=16)
        docs_list = []
        for doc in docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list


from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.rag import retriever

class RAGSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    query : str = Field(description="The query.")


class RAGTool(BaseTool):
    name = "rag"
    description = "Ask any about spatio-temporal knowledge to help you better solve the problem."
    args_schema: Type[RAGSchema] = RAGSchema

    def _run(
            self,
            query : str
    ) -> int:
        """Use the tool."""
        return retriever.query(retriever.get_db(), query)
        
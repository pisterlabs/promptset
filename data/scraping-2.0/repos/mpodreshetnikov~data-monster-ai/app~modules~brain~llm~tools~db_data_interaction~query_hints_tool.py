from pydantic import Field

import yaml

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.vectorstores import Chroma, VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.tools import BaseTool
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit


class SQLQueryHint:
    question: str = Field()
    tables: list[str] = Field()
    query: str = Field()

    def __init__(self, **entries):
        self.__dict__.update(entries)


class SQLQueryHintsToolkit(BaseToolkit):
    queries_path: str = Field(description="Path to the YAML file with the query examples")
    embed_model: Embeddings = Field(exclude=True)
    persist_directory: str = Field(None, description="Path to the directory where the vector store is persisted")
    
    queries: dict = None
    vector_store: VectorStore = None

    def build(self):
        self.vector_store = Chroma(
            collection_name="sql_query_hints",
            embedding_function=self.embed_model,
            persist_directory=self.persist_directory)

        with open(self.queries_path, "r", encoding="utf-8") as f:
            self.queries = yaml.safe_load(f)["list"]
        docs = list(map(lambda x: Document(
                page_content=x["question"],
                metadata={ "cls": yaml.dump(x) }),
            self.queries))

        # TODO remove old collection if persist_directory is set
        self.vector_store.add_documents(docs)

    def get_tools(self) -> list[BaseTool]:
        # TODO change the whole method...
        json_spec = JsonSpec(dict_= { "examples": self.queries }, max_value_length=500)
        json_toolkit = JsonToolkit(spec=json_spec)
        return json_toolkit.get_tools()
    
    def get_top_hints(self, question: str, limit: int) -> list[SQLQueryHint]:
        data: list[Document] = self.vector_store.search(question, search_type="mmr", k=limit)
        hints = list(map(lambda x: SQLQueryHint(**yaml.safe_load(x.metadata["cls"])), data))
        return hints
    
    class Config:
        arbitrary_types_allowed = True

from langchain.vectorstores import VectorStore, ClickhouseSettings

from lively_paper.model.embeddings import m3e
from lively_paper.vector.clickhouse import ClickhousePro


settings = ClickhouseSettings(metric='euclidean')
vector_store: VectorStore = ClickhousePro(embedding=m3e, config=settings)

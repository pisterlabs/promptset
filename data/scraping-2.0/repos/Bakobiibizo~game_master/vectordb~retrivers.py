from langchain.retrievers import (
    SelfQueryRetriever,
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from enum import Enum
import loguru
from vectordb.selector import BaseSelector

logger = loguru.logger


class RetrieverOptions(Enum):
    SELF_QUERY_RETRIEVER = SelfQueryRetriever
    ENSEMBLE_RETRIVER = EnsembleRetriever
    CONTEXTUAL_COMPRESSION_RETRIVER = ContextualCompressionRetriever


class RetrieverSelector(BaseSelector):
    def __init__(self, default_retriever=RetrieverOptions.SELF_QUERY_RETRIEVER):
        super().__init__()
        logger.info("Initializing RetrieverSelector")
        self.self_query_retriever: RetrieverOptions = (
            RetrieverOptions.SELF_QUERY_RETRIEVER
        )
        self.ensemble_retriever: RetrieverOptions = RetrieverOptions.ENSEMBLE_RETRIVER
        self.contextual_compression_retriever: RetrieverOptions = (
            RetrieverOptions.CONTEXTUAL_COMPRESSION_RETRIVER
        )
        self.retriever = default_retriever
        self.initialize_maps(list(RetrieverOptions))

    def get_self_query_retriever(self):
        logger.info("Getting  self_query_retriver")
        return self.select("self_query_retriever")

    def get_ensemble_retriever(self):
        logger.info("Getting  ensemble_retriver")
        return self.select("ensemble_retriever")

    def get_contextual_compression_retriever(self):
        logger.info("Getting  contextual_compression_retriver")
        return self.select("contextual_compression_retriever")

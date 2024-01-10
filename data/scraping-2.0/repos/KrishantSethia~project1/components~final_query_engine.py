

from components.hybridretriever import HybridRetriever
from components.node_postprocessor import NodePostprocessor
from components.dictionary_of_retrievers import DictionaryOfRetrievers
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import OpenAI
from llama_index import ServiceContext


class FinalQueryEngine():

    def final_query_engine(index, nodes):
        retrievers = DictionaryOfRetrievers.dictionary_of_retrievers(
            index=index, nodes=nodes)
        hybrid_retriever = HybridRetriever(
            vector_retriever=retrievers["vector_retriever"], bm25_retriever=retrievers["bm25_retriever"])

        nodepostprocessor = [NodePostprocessor.nodepostprocessor()]

        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model='gpt-4')
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=nodepostprocessor,
            service_context=service_context


        )

        return query_engine

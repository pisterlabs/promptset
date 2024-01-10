from typing import Any
from llama_index import (
    Document,
    VectorStoreIndex,
    get_response_synthesizer,
    StorageContext,
    load_index_from_storage,
    SummaryIndex,
    ServiceContext,
    set_global_service_context,
)

from llama_index.schema import NodeWithScore

from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-4", temperature=0)

import os
from collections import namedtuple

Doc_item = namedtuple("node_item", ["title", "text"])
Query_response = namedtuple("query_response", ["text", "source_nodes"])

DOC_DIR = "data/md/"
INDEX_DIR = "data/index/"

doc_mapping = {
    "About_NYC.md": "https://housingconnect.nyc.gov/PublicWeb/about-us",
    "Applying_for_Affordable_Housing.md": "https://www.nyc.gov/assets/hpd/downloads/pdfs/services/applying-to-affordable-housing-english.pdf",
    "FAQs.md": "https://housingconnect.nyc.gov/PublicWeb/faq",
    "Learn_how_to_use_Housing_Connect.md": "https://housingconnect.nyc.gov/PublicWeb/about-us/training",
    "NYC_gov_policy.md": "http://www.nyc.gov/privacy ",
    "Terms_of_use.md": "https://www.nyc.gov/home/terms-of-use.page",
    "Welcome_to_NYC_Housing_Connect.md": "https://housingconnect.nyc.gov/PublicWeb",
}


class VectorIndexCreator:
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self) -> Any:
        text_list = self.read_files()
        self.create_and_save_index(text_list)
        pass

    def read_files(self) -> None:
        text_list = []
        for k, _ in doc_mapping.items():
            with open(os.path.join(DOC_DIR, k), "r") as f:
                text_list.append(Doc_item(k, f.read()))

        return text_list

    def create_and_save_index(self, text_list: list[Doc_item]) -> VectorStoreIndex:
        documents = [
            Document(doc_id=doc_mapping[doc_item.title], text=doc_item.text)
            for doc_item in text_list
        ]
        parser = SimpleNodeParser(
            text_splitter=SentenceSplitter(chunk_size=512, chunk_overlap=64)
        )
        nodes = parser.get_nodes_from_documents(documents)

        service_context = ServiceContext.from_defaults(llm=llm)

        index = VectorStoreIndex(nodes=nodes, service_context=service_context)
        index.storage_context.persist(persist_dir=INDEX_DIR)

        # return index


class QueryIndex:
    def __init__(self) -> None:
        self.index = self.retrieve_index()
        pass

    def retrieve_index(self) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = load_index_from_storage(
            storage_context=storage_context, service_context=service_context
        )
        return index

    def query_index(self, query_str: str) -> list[NodeWithScore]:
        index = self.index

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
        )

        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)],
        )

        response = query_engine.query(query_str)
        return response

    def query(self, query_str: str) -> Query_response:
        response = self.query_index(query_str)
        # respond with text and source nodes
        response_text = str(response)
        response_sources = list(
            set(
                [
                    node_with_score.node.ref_doc_id
                    for node_with_score in response.source_nodes
                ]
            )
        )

        return Query_response(response_text, response_sources)


if __name__ == "__main__":
    # builder = VectorIndexCreator()
    # builder()
    q_engine = QueryIndex()
    while True:
        query = input("Enter query: ")
        print(q_engine.query(query))

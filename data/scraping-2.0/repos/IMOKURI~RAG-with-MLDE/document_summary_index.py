from llama_index import (
    DocumentSummaryIndex,
    OpenAIEmbedding,
    PromptHelper,
    ServiceContext,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.indices.document_summary import DocumentSummaryIndexLLMRetriever
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.text_splitter import TokenTextSplitter

import prompt_template as pt


class CustomDocumentSummaryIndex:
    def __init__(
        self, openai_api_key: str = "dummy", openai_api_base: str = "http://localhost:8000/v1", streaming: bool = True
    ):
        embed_model = OpenAIEmbedding(embed_batch_size=1, api_key=openai_api_key, api_base=openai_api_base)
        self.llm = OpenAI(
            temperature=0,
            batch_size=1,
            max_tokens=1024,
            api_key=openai_api_key,
            api_base=openai_api_base,
            streaming=streaming,
        )

        text_splitter = TokenTextSplitter(
            separator="。", chunk_size=4096, chunk_overlap=64, backup_separators=["、", " ", "\n"]
        )
        node_parser = SimpleNodeParser(text_splitter=text_splitter)
        prompt_helper = PromptHelper(
            context_window=4096, num_output=1024, chunk_overlap_ratio=0.05, chunk_size_limit=None, separator="。"
        )

        self.service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper
        )

        self.response_synthesizer = get_response_synthesizer(
            service_context=self.service_context,
            text_qa_template=pt.CHAT_TEXT_QA_PROMPT,
            summary_template=pt.CHAT_TREE_SUMMARIZE_PROMPT,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            use_async=True,
            streaming=streaming,
        )

        self.index = None
        self.retriever = None
        self.query_engine = None

    def from_documents(self, documents):
        self.index = DocumentSummaryIndex.from_documents(
            documents,
            service_context=self.service_context,
            response_synthesizer=self.response_synthesizer,
            summary_query=pt.SUMMARY_QUERY,
            show_progress=True,
            embed_summaries=False,
        )

    def persist(self, path: str = "rag-system"):
        assert self.index is not None
        self.index.storage_context.persist(path)

    def load(self, path: str = "rag-system"):
        storage_context = StorageContext.from_defaults(persist_dir=path)
        self.index = load_index_from_storage(storage_context)

    def get_summary(self, doc_id: str):
        assert self.index is not None
        return self.index.get_document_summary(doc_id)

    def as_retriever(self, top_k: int = 1):
        assert self.index is not None
        self.retriever = DocumentSummaryIndexLLMRetriever(
            self.index,
            choice_select_prompt=pt.DEFAULT_CHOICE_SELECT_PROMPT,
            # choice_batch_size=10,
            choice_top_k=top_k,
            service_context=self.service_context,
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )

    def retrieve(self, query: str):
        assert self.retriever is not None
        return self.retriever.retrieve(query)

    def query(self, query: str):
        assert self.query_engine is not None
        return self.query_engine.query(query)

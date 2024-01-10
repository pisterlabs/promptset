import logging
import os
from typing import List

from langchain.agents import Tool
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    LLMPredictor,
    OpenAIEmbedding,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StringIterableReader,
    VectorStoreIndex,
)
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import RedisVectorStore

from expert_gpts.embeddings.base import EmbeddingsHandlerBase
from shared.config import EMBEDDINGS_TYPE
from shared.llm_manager_base import BaseLLMManager
from shared.llms.openai import GPT_3_5_TURBO
from shared.llms.system_prompts import (
    GET_MEMORIES_TOOL_PROMPT,
    SAVE_MEMORIES_TOOL_PROMPT,
)

logger = logging.getLogger(__name__)

EMBEDS_MODEL = OpenAIEmbedding()
embeddings = OpenAIEmbeddings()


# https://zeeshankhawar.medium.com/connecting-chatgpt-with-your-own-data-using-llama-index-and-langchain-74ba79fb7429
# https://betterprogramming.pub/llamaindex-how-to-use-index-correctly-6f928b8944c6
# https://medium.com/badal-io/exploring-langchain-and-llamaindex-to-achieve-standardization-and-interoperability-in
# -large-2b5f3fabc360
# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/
# https://dev.to/iamadhee/combine-langchain-llama-index-1068
# https://cobusgreyling.medium.com/llamaindex-chat-engine-858311dfb8cb


class LlamaIndexEmbeddingsHandler(EmbeddingsHandlerBase):
    _instances = {}

    def __init__(
        self,
        llm_manager: BaseLLMManager,
        embeddings: EMBEDDINGS_TYPE,
        index_name: str = "pg_essays",
        index_prefix: str = "llama",
        load_docs: bool = False,
    ):
        vector_store = RedisVectorStore(
            index_name=index_name,
            index_prefix=index_prefix,
            redis_url=os.getenv("REDIS_URL"),
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        llm_predictor = LLMPredictor(
            llm=llm_manager.get_llm(
                max_tokens=256, temperature=0.9, model=GPT_3_5_TURBO, as_predictor=True
            )
        )
        node_parser = SimpleNodeParser(
            text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
        )
        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=EMBEDS_MODEL,
            node_parser=node_parser,
            prompt_helper=prompt_helper,
        )
        if not load_docs:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                service_context=service_context,
            )
        else:
            self.document_strings = []
            self.documents = []
            for key, value in embeddings.items():
                if value.folder_path:
                    self.documents.extend(
                        SimpleDirectoryReader(
                            value.folder_path, recursive=True
                        ).load_data()
                    )

            for key, value in embeddings.items():
                if value.content:
                    self.document_strings.append(value.content)
            self.documents.extend(
                StringIterableReader().load_data(self.document_strings)
            )

            self.index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                service_context=service_context,
            )

        self.metadata_postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls_key not in cls._instances:
            cls_key = (
                f"llamaindex_instance_{kwargs['index_name']}_{kwargs['index_prefix']}"
            )

            cls._instances[cls_key] = cls(*args, **kwargs)

        return cls._instances[cls_key]

    def search(self, query: str) -> RESPONSE_TYPE:
        logger.debug(f"query: {query}")
        # https://gpt-index.readthedocs.io/en/latest/examples/node_postprocessor/MetadataReplacementDemo.html
        return self.index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[self.metadata_postprocessor],
        ).query(query)

    def save(self, remember_this: List[str]):
        logger.debug(f"remember_this: {remember_this}")
        documents = StringIterableReader().load_data(remember_this)
        self.documents.extend(documents)
        for document in documents:
            self.index.insert(document)

    def get_embeddings_tool_get_memory(self, tool_key: str = "default") -> Tool:
        return Tool(
            name=f"{tool_key}_get_memories",
            func=lambda q: str(self.search(q)),
            description=GET_MEMORIES_TOOL_PROMPT,
        )

    def get_embeddings_tool_save_memory(self, tool_key: str = "default") -> Tool:
        return Tool(
            name=f"{tool_key}_save_memory",
            func=lambda q: str(self.save([q])),
            description=SAVE_MEMORIES_TOOL_PROMPT,
        )

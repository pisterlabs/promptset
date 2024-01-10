import os, sys, json
from .embedding import get_embeddings
from .llm import get_model
from .webscrape import scrape_site
from typing import List
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Qdrant
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import WikipediaRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.utils import DistanceStrategy
from .utils import import_from_path


class Gerty:
    def __init__(
        self,
        n_ctx: int = 4096,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "models", "nous-hermes-llama-2-7b"
        ),
    ):
        self.n_ctx = n_ctx
        self.model_path = os.path.realpath(model_path)

        self.embedding_model = get_embeddings(model_path=model_path, n_ctx=self.n_ctx)
        self.language_model = get_model(model_path=model_path, n_ctx=self.n_ctx)

        self.prompts_import = import_from_path(
            os.path.realpath(os.path.join(self.model_path, "prompts.py")),
            "prompts"
        )

        self.prompt_template = self.prompts_import.DEFAULT_PROMPT_TEMPLATE
        self.prompt_variables = self.prompts_import.DEFAULT_PROMPT_VARIABLES
        self.retrievers = []
        self.tools = None
        self.final_retriever = None

    def get_stuff_prompt(self):
        return PromptTemplate(
            input_variables=self.prompt_variables, template=self.prompt_template
        )

    # def get_map_reduce_prompt(self):
    #    return {
    #            "question_prompt": self.get_stuff_prompt(),
    #            "combine_prompt": PromptTemplate(input_variables=DEFAULT_CONDENSE_PROMPT_VARIABLES,
    #                                             template=DEFAULT_CONDENSE_PROMPT_TEMPLATE)
    #    }

    def get_qa_model(self, memory=None):
        chain_type_kwargs = {"prompt": self.get_stuff_prompt()}
        # chain_type_kwargs = self.get_map_reduce_prompt()
        return self.__get_conversational_model(chain_type_kwargs, memory=memory)

    def __get_retrievers__(self):
        if len(self.retrievers) == 0 or self.final_retriever is None:
            self.retrievers = []

            # db_retriever = self.db.as_retriever(
            #    search_type="similarity_score_threshold",
            #    search_kwargs={"score_threshold": 0.5},
            # )
            db_retriever = self.db.as_retriever(search_type="mmr")
            # SelfQueryRetriever.from_llm(
            #    llm=self.language_model,
            #    vectorstore=self.db,
            #    document_contents="Information about the distributed compute protocol (DCP) and it's javascript api/spec",
            #    metadata_field_info=None,
            #    verbose=True,
            # )

            self.retrievers.append(db_retriever)
            # self.retrievers.append(WikipediaRetriever(load_max_docs=1))
            merger_retriever = EnsembleRetriever(
                retrievers=self.retrievers,
                weights=[1 / len(self.retrievers)] * len(self.retrievers),
            )
            # MergerRetriever(retrievers=self.retrievers)
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embedding_model, similarity_threshold=0.76
            )
            # reordering = LongContextReorder()
            compressor = LLMChainExtractor.from_llm(self.language_model)
            filter_pipeline = DocumentCompressorPipeline(
                transformers=[embeddings_filter, compressor]
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=filter_pipeline, base_retriever=merger_retriever
            )
            self.final_retriever = (
                compression_retriever  # merger_retriever  # compression_retriever
            )
        return self.final_retriever

    def __get_conversational_model(self, chain_type_kwargs, memory=None):
        if memory is None:
            memory = ConversationSummaryBufferMemory(
                llm=self.language_model,
                memory_key="chat_history",
                ai_prefix="Gerty",
                return_messages=True,
                max_token_limit=1024,
                prompt=PromptTemplate(
                    input_variables=self.prompts_import.DEFAULT_CONVO_SUMMARY_VARIABLES,
                    template=self.prompts_import.DEFAULT_CONVO_SUMMARY_TEMPLATE,
                ),
            )
        doc_chain = load_qa_chain(
            self.language_model,
            chain_type="stuff",
            verbose=True,
            callbacks=None,
            **chain_type_kwargs
        )

        condense_question_chain = NoOpLLMChain(self.language_model)

        return ConversationalRetrievalChain(
            retriever=self.__get_retrievers__(),
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=None,
            memory=memory,
        )

    def embed_db(self, knowledge_base, cache_dir, url=False, chunk_size=1000, **kwargs):
        loaders = []

        if url:
            _kbs = []
            for kb in knowledge_base:
                _kbs = scrape_site(kb, _kbs)
            knowledge_base = _kbs

            loader = WebBaseLoader(knowledge_base)
        else:
            for kb in knowledge_base:
                loader = DirectoryLoader(kb, glob=kwargs["glob"])
                loaders.append(loader)
            loader = MergedDataLoader(loaders=loaders)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=20
        )
        docs = text_splitter.split_documents(documents)

        db_name = input("What is the name of this db? ")
        db_name = db_name.replace("\n", "").replace(" ", "-")
        if cache_dir:
            self.db_cache_dir = cache_dir
        self.db = Qdrant.from_documents(
            docs, self.embedding_model, path=self.db_cache_dir, collection_name=db_name
        )

        return

    def load_db(self, cache_dir=None):
        if cache_dir:
            self.db_cache_dir = cache_dir
        import qdrant_client

        client = qdrant_client.QdrantClient(
            path=self.db_cache_dir,
        )
        db_name = input("What is the name of this db? ")
        db_name = db_name.replace("\n", "").replace(" ", "-")
        self.db = Qdrant(
            client=client,
            collection_name=db_name,
            embeddings=self.embedding_model,
        )
        return


class NoOpLLMChain(LLMChain):
    """No-op LLM chain."""

    def __init__(self, llm):
        super().__init__(
            llm=llm, prompt=PromptTemplate(template="", input_variables=[])
        )

    async def arun(self, question: str, *args, **kwargs) -> str:
        return question

    def run(self, question: str, *args, **kwargs) -> str:
        return question

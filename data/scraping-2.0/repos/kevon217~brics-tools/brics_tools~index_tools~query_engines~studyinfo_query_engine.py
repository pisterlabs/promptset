import os
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    LangchainEmbedding,
    StorageContext,
    QueryBundle,
    set_global_service_context,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms import OpenAI
import tiktoken
import chromadb
from llama_index.schema import MetadataMode
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler
from llama_index.prompts import PromptTemplate

# from brics_tools.index_tools.index_managers.studyinfo_summary_manager import (
#     StudyInfoSummaryIndexManager,
# )
from brics_tools.index_tools.index_managers.studyinfo_vectorstore_manager import (
    StudyInfoVectorStoreIndexManager,
)

from brics_tools.index_tools.node_postprocessors.studyinfo_node_postprocessor import (
    TopNForLLMSynthesisNodePostprocessor,
)
from brics_tools.index_tools.prompts.studyinfo_prompts import STUDYINFO_QA_PROMPT
from brics_tools.index_tools.service_contexts.api_key_context_manager import (
    use_openai_api_key,
)

from brics_tools.index_tools.query_engines import logger, log, copy_log


class StudyInfoQueryEngine:
    def __init__(self, config):
        self.config = config
        self.current_engine = None  # Initialize current_engine as None
        logger.info("Initializing StudyInfoQueryEngine")
        self.init_callback_manager()

    def init_callback_manager(self):  # TODO: figure out unified solution for this
        logger.info("Initializing Callback Manager")
        self.callback_handler = OpenInferenceCallbackHandler()
        self.callback_manager = CallbackManager([self.callback_handler])

    def init_llm(self, model_name=None, temperature=None, api_key=None):
        if os.environ.get("OPENAI_API_KEY", ""):
            if not model_name:
                model_name = (
                    self.config.query_engines.studyinfo_query_engine.llm.llm_kwargs.model_name
                )
                temperature = (
                    self.config.query_engines.studyinfo_query_engine.llm.llm_kwargs.temperature
                )
            self.llm = OpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            logger.info(
                f"Initializing LLM: model_name={model_name}, temperature={temperature}"
            )
        else:
            logger.info("No OpenAI API key provided. LLM will not be used.")
            self.llm = None  # No LLM operations can

    def init_service_context(self):
        if self.llm:
            logger.info("Initializing Service Context with LLM")

            self.service_context = ServiceContext.from_defaults(llm=self.llm)
        else:
            logger.info("Initializing Service Context without LLM")
            self.service_context = ServiceContext.from_defaults(llm=None)

    def init_vector_index(self):
        logger.info("Initializing Vector Index")
        vector_index_manager = StudyInfoVectorStoreIndexManager(
            self.config.index_managers.studyinfo_vectorstore_index
        )
        vector_index_manager.load_vectorstore_index()
        self.vector_index_manager = vector_index_manager
        index = vector_index_manager.index
        self.vector_index = index
        # return index

    def init_vector_retriever(self, similarity_top_k=None):
        similarity_top_k = (
            similarity_top_k
            or self.config.query_engines.studyinfo_query_engine.retrievers.VectorIndexRetriever.similarity_top_k
        )
        logger.info(
            f"Initializing Vector Retriever with similarity_top_k = {similarity_top_k}"
        )
        retriever = VectorIndexRetriever(
            index=self.vector_index, similarity_top_k=similarity_top_k
        )
        self.retriever = retriever
        # return retriever

    def init_node_postprocessors(self, rerank_top_n=None, top_n_for_llm=None):
        rerank_top_n = (
            rerank_top_n
            or self.config.query_engines.studyinfo_query_engine.node_postprocessors.rerank.cross_encoder.top_n
        )
        logger.info(
            f"Initializing Node Postprocessors with rerank_top_n = {rerank_top_n}"
        )
        top_n_for_llm = (
            top_n_for_llm
            or self.config.query_engines.studyinfo_query_engine.node_postprocessors.limit.top_n
        )
        logger.info(
            f"Initializing Node Postprocessors with top_n_for_llm = {top_n_for_llm}"
        )
        node_postprocessors = [
            SentenceTransformerRerank(
                model=self.config.query_engines.studyinfo_query_engine.node_postprocessors.rerank.cross_encoder.model_name,
                top_n=rerank_top_n,
            ),
            TopNForLLMSynthesisNodePostprocessor(
                rerank_top_n, top_n_for_llm=top_n_for_llm
            ),
        ]  # TODO: add classes to config for more flexibility
        self.node_postprocessors = node_postprocessors
        # return node_postprocessors

    def postprocess(self, nodes, user_query):
        logger.info("Postprocessing nodes")
        query_bundle = QueryBundle(user_query)
        for node_postprocessor in self.node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes=nodes, query_bundle=query_bundle
            )
        return nodes

    def init_response_synthesizer(self, response_mode, text_qa_template=None):
        response_mode = (
            response_mode
            if response_mode is not None
            else self.config.query_engines.studyinfo_query_engine.response_synthesizer.response_mode
        )
        # response_mode = self.config.query_engines.studyinfo_query_engine.response_synthesizer.response_mode
        if text_qa_template is None:
            text_qa_template = STUDYINFO_QA_PROMPT
        else:
            text_qa_template = PromptTemplate(
                text_qa_template, prompt_type="text_qa"
            )  # self.config.query_engines.studyinfo_query_engine.response_synthesizer.text_qa_template
        # refine_template = # self.config.query_engines.studyinfo_query_engine.response_synthesizer.refine_template
        # self.response_mode = response_mode
        service_context = (
            self.service_context if hasattr(self, "service_context") else None
        )
        logger.info(
            f"Initializing Response Synthesizer: response_mode = {response_mode}"
        )
        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            response_mode=response_mode,
            text_qa_template=text_qa_template,
            callback_manager=self.callback_manager,
        )
        self.response_synthesizer = response_synthesizer
        # return response_synthesizer

    def create_retriever_only_engine(
        self,
        similarity_top_k=None,
        rerank_top_n=None,
        response_mode="no_text",
        text_qa_template=None,
    ):
        logger.info("Creating Retriever-only Engine")
        self.init_vector_retriever(similarity_top_k=similarity_top_k)
        self.init_node_postprocessors(
            rerank_top_n=rerank_top_n, top_n_for_llm=rerank_top_n
        )  # #HACK: setting to rerank_top_n since no need for llm
        # synthesizer = self.init_response_synthesizer(response_mode=response_mode,text_qa_template=text_qa_template)

        retriever_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,
            node_postprocessors=self.node_postprocessors,
            callback_manager=self.callback_manager,  # Or however you manage this
        )
        self.retriever_engine = retriever_engine
        # self.init_node_postprocessors(rerank_top_n=rerank_top_n)
        # return retriever_engine

    def create_retriever_query_engine(
        self,
        model_name=None,
        temperature=None,
        similarity_top_k=None,
        rerank_top_n=None,
        top_n_for_llm=None,
        response_mode=None,
        text_qa_template=None,
    ):
        logger.info("Creating Retriever Query Engine")
        self.init_llm(model_name=model_name, temperature=temperature)
        self.init_service_context()
        self.init_vector_retriever(similarity_top_k=similarity_top_k)
        self.init_node_postprocessors(
            rerank_top_n=rerank_top_n, top_n_for_llm=top_n_for_llm
        )
        self.init_response_synthesizer(
            response_mode=response_mode, text_qa_template=text_qa_template
        )

        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=self.node_postprocessors,
            callback_manager=self.callback_manager,  # Or however you manage this
        )
        self.query_engine = query_engine
        # return query_engine

    # def init_summary_index(self):
    #     # Initialize StudyInfoSummaryIndexManager and load the summary index
    #     self.summary_index_manager = StudyInfoSummaryIndexManager(self.config.index_managers.studyinfo_summary_index)
    #     self.summary_index_manager.load_summary_index()
    #     self.summary_index = self.summary_index_manager.index  # Assuming the loaded index is stored in 'index' attribute

    # def init_document_summary_retriever(self, similarity_top_k=None):
    #     similarity_top_k = similarity_top_k or self.config.query_engines.studyinfo_query_engine.retrievers.DocumentSummaryIndexEmbeddingRetriever.similarity_top_k
    #     self.retrievers['DocumentSummaryIndexEmbeddingRetriever'] = DocumentSummaryIndexEmbeddingRetriever(
    #         index=self.summary_index,
    #         similarity_top_k=similarity_top_k
    #     )

    # def init_retrievers(self):
    #     self.init_vector_index_retriever()
    #     self.init_document_summary_retriever()

    # def init_summary_query_engine(self):
    #     # assemble query engine
    #     self.query_engines['summary_query_engine'] = RetrieverQueryEngine(
    #         retriever=self.retrievers['DocumentSummaryIndexEmbeddingRetriever'],
    #         response_synthesizer=self.response_synthesizer,
    #         node_postprocessors=self.node_postprocessors,
    #         callback_manager=self.summary_index_manager.callback_manager
    #     )

    # def init_llm_model(self):
    #     #TODO: figure out where this is needed
    #     llm_model_name = self.query_engine_config.llm.llm_kwargs.model_name
    #     temperature = self.query_engine_config.llm.llm_kwargs.temperature
    #     self.llm = OpenAI(temperature=temperature, model=llm_model_name)

    # # Initialize RetrieverQueryEngine
    # self.query_engine = RetrieverQueryEngine(
    #     retriever=self.retriever,
    #     response_synthesizer=self.response_synthesizer
    # )

    # def init_query_engine_tools(self):
    #     # TODO: logic to init query engines if not already initialized
    #     self.init_query_engines()
    #     self.summary_tool = QueryEngineTool.from_defaults(query_engine=self.summary_query_engine,
    #                                                       description="Userful for retrieving high-level summaries of research studies based on query context.",)

    #     self.vector_tool = QueryEngineTool.from_defaults(query_engine=self.vector_query_engine,description="Useful for performing deep, context-aware search to find research studies closely related to specific queries.",)

    # def init_router_query_engine(self):
    #     # Initializ]=e a RouterQueryEngine to use multiple retrievers
    #     self.router_query_engine = RouterQueryEngine(selector=PydanticSingleSelector.from_defaults(),   query_engine_tools=[self.summary_tool,
    #                                                                                                             self.vector_tool, ],)

    # def display_results(self, response):
    #     # Display the results in a readable format
    #     # This can be customized further
    #     print(f"Query Results: {response}")
    #     print(f"Source Nodes: {response.source_nodes}")

    # def results_to_dataframe(self, response):
    #     # Convert the results to a dataframe
    #     # This can be customized further
    #     df = response.to_dataframe()
    #     return df
    # def export_results_to_csv(self, response, filepath):
    #     df = self.results_to_dataframe(response)
    #     df.to_csv(filepath)


# def visualize_retrieved_nodes(nodes) -> None:
#     result_dicts = []
#     for node in nodes:
#         node = deepcopy(node)
#         node.node.metadata = None
#         node_text = node.node.get_text()
#         node_text = node_text.replace("\n", " ")

#         result_dict = {"Score": node.score, "Text": node_text}
#         result_dicts.append(result_dict)

#     df = pd.DataFrame(result_dicts)

#     return df`        ?????????????.?:/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;/;lklllllllllllllllln l l]'[ yh h hhn  ]

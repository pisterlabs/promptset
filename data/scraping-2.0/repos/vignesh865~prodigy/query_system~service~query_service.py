import logging

from langchain.chains import RetrievalQA
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever

from source_consumer.service.collection_strategy import CollectionStrategy
from source_consumer.service.vector_orchestrator import VectorOrchestrator


class QueryService:

    # ### Retriever

    # #### MultiQueryRetriever
    def get_multiquery_retriever(self, vector_ref):
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        return MultiQueryRetriever.from_llm(
            retriever=vector_ref.as_retriever(), llm=self.llm
        )

    def get_ensemble_retriever(self, base_llm_retriever, documents):
        txts = [i.page_content for i in documents]
        bm25_retriever = BM25Retriever.from_texts(txts)
        bm25_retriever.k = 2
        return EnsembleRetriever(retrievers=[bm25_retriever, base_llm_retriever], weights=[0.8, 0.2])

    def get_compression_retriever(self, base_retriever):
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, redundant_filter]
        )
        return ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)

    def get_retriever(self, vector_ref, documents):
        multiquery_retriever = self.get_multiquery_retriever(vector_ref)
        ensemble_retriever = self.get_ensemble_retriever(multiquery_retriever, documents)
        return self.get_compression_retriever(ensemble_retriever)

    def prepare_chain(self, retriever):
        template = """Use the following pieces of context to answer the question at the end. 
        If yoau don't know the answer, just say that you don't know, don't try to make up an answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            #     chain_type="map_reduce",
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, }
        )

    def init_prodigy(self, tenant_id):

        collection_name = CollectionStrategy().get_collection_name_by_tenant(tenant_id)

        vo = VectorOrchestrator()

        embedding_model, vector_size = vo.get_embeddings()
        vector_store = vo.get_vector_store(embedding_model, vector_size)

        vector_ref = vector_store.get_store(collection_name)
        documents = vector_store.get_documents(collection_name)

        retriever = self.get_retriever(vector_ref, documents)

        self.chain = self.prepare_chain(retriever)

        return self

    __instance = {}

    def __init__(self, tenant_id):
        """ Virtually private constructor. """
        if QueryService.__instance.get(tenant_id) is not None:
            raise Exception("This class is a singleton!")
        else:

            self.embeddings = SentenceTransformerEmbeddings()
            # self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.llm = Ollama(temperature=0, model="mistral")
            self.chain = None

            self.init_prodigy(tenant_id)

            QueryService.__instance[tenant_id] = self

    @staticmethod
    def get_instance(tenant_id):
        """ Static access method. """
        if QueryService.__instance.get(tenant_id) is None:
            QueryService(tenant_id)

        return QueryService.__instance.get(tenant_id)

    def answer_with_chain(self, question):

        if self.chain is None:
            raise ValueError("Chain is not instantiated")

        return self.chain({"query": question.lower()})

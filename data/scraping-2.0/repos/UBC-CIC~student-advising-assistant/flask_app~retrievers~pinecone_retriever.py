from langchain.schema import Document
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
import pinecone 
from typing import List, Dict, Optional, Tuple
import os
import ast
import copy
from .tools import load_json_file
from .base import Retriever, INDEX_PATH

class MyPineconeRetriever(PineconeHybridSearchRetriever):
    """
    Wrapper of LangChain PineconeHybridSearchRetriever
    that allows for additional parameters in query,
    and fetching documents by ID
    """
    # Key for the original text in the pinecone document's metadata
    _text_key: str = 'context'
    # Key to place the document similarity score in metadata after retrieval
    _score_key: str = 'score'
    
    def _handle_pinecone_docs(self, vectors: List[dict]) -> List[Document]:
        """
        Convert a list of vectors from a pinecone query or fetch to a list
        of Langchain documents
        """                
        docs = []
        for res in vectors:
            context = res["metadata"].pop(self._text_key)
            doc = Document(page_content=context, metadata=res["metadata"])
            if self._score_key in res: doc.metadata[self._score_key] = res["score"]
            docs.append(doc)
        return docs
        
    def _get_relevant_documents(
        self, 
        query: str, 
        **kwargs
    ) -> List[Document]:
        """
        Adapted from langchain.retrievers.pinecone_hybrid_search, with support for
        additional keyword arguments to the pinecone query.
        Example keyword arguments
        - namespace: pinecone namespace to search in
        - filter: metadata filter to apply in the pinecone call, see
                           https://docs.pinecone.io/docs/metadata-filtering
        The similarity score of each document is placed in the document metadata
        under the 'score' key.
        """
        
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        print(kwargs['filter'])
        response = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            **kwargs
        )
        return self._handle_pinecone_docs(response["matches"])
    
    def fetch_by_id(self, ids: List[int], namespace: Optional[str] = None):
        """
        Fetch a set of documents by ids
        """
        ids = [str(id) for id in ids]
        response = self.index.fetch(ids=ids, namespace=namespace)
        return self._handle_pinecone_docs(response['vectors'].values())
    
class PineconeRetriever(Retriever):
    index_type: str = 'pinecone'
    index_config_path: str = os.path.join(INDEX_PATH, index_type, 'index_config.json')
    bm25_weights_path: str = os.path.join(INDEX_PATH, index_type, 'bm25_params.json')
    
    # Parameters from the program_info to use in the metadata filter for queries
    filter_params: List[str]
    # Namespace to use for all queries to the pinecone index
    namespace: Optional[str] 
    
    def __init__(self, pinecone_key: str, pinecone_region: str, alpha = 0.4, verbose: bool = False):
        """
        Initialize the pinecone retriever
        - pinecone_key: API key for pinecone
        - pinecone_region: region for the pinecone index
        - alpha: weighting of the sparse vs dense vectors
                 0 = pure semantic search (dense vectors)
                 1 = pure keyword search (sparse vectors)
        """
        super().__init__(verbose)
        
        # Load the config file
        index_config = load_json_file(self.index_config_path)
        
        # Load the sparse vector model
        bm25_encoder = BM25Encoder().load(self.bm25_weights_path)
        
        # Load the dense embedding model
        embeddings_model = self._embeddings_model_from_config(index_config)
        self.num_embed_concats = len(index_config['embeddings'])
        
        # Connect to the pinecone index
        pinecone.init(      
            api_key=pinecone_key,      
            environment=pinecone_region     
        )     
        index = pinecone.Index(index_config['name'])
        self.namespace = index_config['namespace']
        
        # Create the retriever
        hybrid_retriever = MyPineconeRetriever(
            embeddings=embeddings_model, sparse_encoder=bm25_encoder, index=index
        )
        hybrid_retriever.alpha = alpha
        self.retriever = hybrid_retriever
    
    def semantic_search(self, filter: Dict, program_info: Dict, topic: str, query: str, k = 5, threshold = 0) -> List[Document]:
        """
        Return the documents from similarity search with the given context and query
        - filter: Dict of metadata filter keys and values
        - program_info: Dict of program information
        - topic: keyword topic of the query
        - query: the full query question
        - k: number of documents to return
        - threshold: relevance threshold, all returned documents must surpass the threshold
                     relevance is dot-product based, so not normalized
                     larger scores indicate greater relevance
        """
        self.set_top_k(k)
        query_str, kwargs = self._query_converter(filter,program_info,topic,query)
        if self.namespace: kwargs['namespace'] = self.namespace
        
        self._output_query_verbose(query_str, kwargs)
        docs = self.retriever.get_relevant_documents(query_str,**kwargs)
        
        docs = self._apply_score_threshold(docs, threshold)
        return self._response_converter(docs)
    
    def docs_from_ids(self, doc_ids: List[int]) -> List[Document]:
        """
        Return a list of documents from a list of document indexes
        """
        docs = self.retriever.fetch_by_id(doc_ids, self.namespace)
        return self._response_converter(docs)
    
    def set_top_k(self, k: int):
        """
        Set the retriever's 'top k' parameter, determines how many
        documents to return from semantic search
        """
        self.retriever.top_k = k
        
    def _query_converter(self, filter: Dict, program_info: Dict, topic: str, query: str) -> Tuple[str,Dict]:
        """
        Generates a text query and keyword args for the retriever from the input
        - filter: Dict of metadata filter keys and values
        - program_info: Dict of program information
        - topic: keyword topic of the query
        - query: the full query question
        Returns:
        - Tuple of the query string, and Dict of metadata to pass to the retriever
        """
        retriever_filter = {}
        for key,val in filter.items():
            retriever_filter[key] = {"$eq": val}

        query_str = ' : '.join([value for value in list(program_info.values()) + [topic,query] if len(value) > 0])
        return self._retriever_combined_query(query_str), {'filter': retriever_filter}
    
    def _response_converter(self, response: List[Document]) -> List[Document]:
        """
        Decode the document metadatas from pinecone
        Since pinecone requires primitive datatypes for metadatas,
        evaluates strings into dicts/arrays
        """
        decode_columns = ['titles','parent_titles','links']
        for doc in response:
            for column in decode_columns:
                doc.metadata[column] = ast.literal_eval(doc.metadata[column])
            doc.page_content = doc.metadata['text']

        return response
    
    def _apply_score_threshold(self, docs: List[Document], threshold) -> List[Document]:
        """
        Filters out documents that do not meet the similarity score threshold
        Assumes the similarity score is in document metadata 'score' key
        - docs: List of documents to filter
        - threshold: relevance threshold, all returned documents must surpass the threshold
                     relevance is in the range [0,1] where 0 is dissimilar, and 1 is most similar
        """
        return [doc for doc in docs if doc.metadata['score'] >= threshold]
        
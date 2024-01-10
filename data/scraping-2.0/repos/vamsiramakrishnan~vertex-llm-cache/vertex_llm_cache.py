import logging
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from typing import List, Any, Dict, Union, Callable, Optional
from functools import wraps
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_connection(func: Callable) -> Callable:
    """
    This decorator ensures that the Elasticsearch connection is established before executing the decorated function.

    :param func: The function to be decorated.
    :return: The decorated function.
    """
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        if not instance.es:
            raise ValueError("Elasticsearch connection not established!")
        return func(instance, *args, **kwargs)
    return wrapper


# Cache Client
class ESCacheClient:
    """
    This class represents an Elasticsearch Cache Client.
    """
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 9200, 
        vector_dims: int = 786, 
        username: str = 'elastic', 
        password:str = 'pass#123', 
        cert_path: str = './instance_ss.crt'):
        """
        Initialize the Elasticsearch Cache Client.

        :param host: The hostname for the Elasticsearch server.
        :param port: The port number for the Elasticsearch server.
        :param vector_dims: The dimensions of the vector embeddings.
        :param username: The username for the Elasticsearch server.
        :param password: The password for the Elasticsearch server.
        :param cert_path: The path to the SSL certificate for the Elasticsearch server.
        """
        
        self.es_config = ESConnectionConfig(host=host, port=port, username=username, password= password, cert_path=cert_path)
        self.es_connection = ESConnection(config=self.es_config, logger=logger)
        self.es_connection.establish_connection()


        self.index_manager = ESIndexManager(self.es_connection)
        self.document_manager = ESDocumentManager(self.es_connection, vector_dims)
        self.search = ESSearch(self.es_connection, vector_dims)

    def setup_index(self, *args, **kwargs):
        """
        Set up the index in Elasticsearch.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The result of the index setup operation.
        """
        return self.index_manager.setup_index(*args, **kwargs)

    def insert_document(self, *args, **kwargs):
        """
        Insert a document into Elasticsearch.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The result of the document insertion operation.
        """
        return self.document_manager.insert_document(*args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        """
        Perform a hybrid search in Elasticsearch.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The result of the hybrid search operation.
        """
        return self.search.hybrid_search(*args, **kwargs)


@dataclass(frozen=True)
class ESConnectionConfig:
    """
    This class represents an immutable configuration for Elasticsearch connection.
    """
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    cert_path: str = './instance_ss.crt'

class ESConnection:
    """
    This class manages Elasticsearch connection.
    """
    def __init__(self, config: ESConnectionConfig, logger: logging.Logger):
        """
        Initialize the Elasticsearch connection.

        :param config: The configuration for the Elasticsearch connection.
        :param logger: The logger instance.
        """
        self.config = config
        self.client = None
        self.logger = logger

    def establish_connection(self) -> None:
        """
        Establishes connection to Elasticsearch.
        """
        try:
            hosts = [{
                'host': self.config.host,
                'port': self.config.port,
                'scheme': 'https',
                'use_ssl': True
            }]
            
            es_connection_params = {
                'hosts': hosts,
                'verify_certs': True,
                'ca_certs': self.config.cert_path
            }

            if self.config.username and self.config.password:
                es_connection_params['http_auth'] = (self.config.username, self.config.password)

            self.client = Elasticsearch(**es_connection_params)

            # Check the connection
            if not self.client.ping():
                raise ValueError("Connection failed!")
            self.logger.info("Connected to Elasticsearch")
        except Exception as e:
            self.logger.error(f"Error establishing connection: {e}")
            raise


# Index Management
class ESIndexManager:
    """
    This class manages Elasticsearch indices.
    """
    def __init__(self, es_connection: ESConnection):
        """
        Initialize the Elasticsearch Index Manager.

        :param es_connection: The Elasticsearch connection instance.
        """
        self.es = es_connection.client
        self.logger = logger

    @ensure_connection
    def setup_index(self, index_name: str, force_deletion: bool = False, vector_dims:int=786, similarity_type: str = "dot_product", knn_type: str = "exact", hnsw_params: dict = None) -> None:
        """
        Set up the index with the necessary mappings for hybrid search.
        If the index already exists and force_deletion is set to True, the index will be deleted and recreated.

        :param index_name: The name of the index to be set up.
        :param force_deletion: If True, deletes the index if it already exists.
        :param similarity_type: The type of similarity metric to use. Options are "cosine", "dot_product", "l2_norm", etc.
        :param knn_type: The type of KNN search. Options are "exact" or "hnsw".
        :param hnsw_params: A dictionary containing hnsw specific parameters like "m" and "ef_construction".
        """

        if knn_type == "hnsw" and hnsw_params is None:
            hnsw_params = {"type": "hnsw", "m": 32, "ef_construction": 100}

        vector_field = {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": similarity_type
        }

        if knn_type == "hnsw":
            vector_field["index_options"] = hnsw_params
            self.knn_type = "hnsw"
        elif knn_type == "exact":
            self.knn_type = "exact"
            # If exact, we don't need to add any specific parameters for the vector field.
            # The script_score query will handle the exact similarity computation.
            pass
        else:
            raise ValueError(f"Invalid knn_type: {knn_type}. Supported values are 'hnsw' and 'exact'.")

        mappings = {
            "mappings": {
                "properties": {
                    "Question": {"type": "text"},
                    "Question_Keywords": {"type": "keyword"},
                    "Answer": {"type": "text"},
                    "Answer_Keywords": {"type": "keyword"},
                    "Question_Vector": vector_field,
                    "Answer_Vector": vector_field
                },
                "_source": {
                    "excludes": ["Question_Vector", "Answer_Vector"]
                }
            }
        }
        try:
            if self.es.indices.exists(index=index_name):
                if force_deletion:
                    self.es.indices.delete(index=index_name)
                    self.logger.info(f"Index '{index_name}' deleted.")
                else:
                    self.logger.warning(f"Index '{index_name}' already exists and force_deletion is not set. Exiting setup.")
                    return

            self.es.indices.create(index=index_name, body=mappings)
            self.logger.info(f"Index '{index_name}' set up successfully.")
        except Exception as e:
            self.logger.error(f"Error setting up index: {e}")
            raise


# Document Management
class ESDocumentManager:
    """
    This class manages Elasticsearch documents.
    """
    def __init__(self, es_connection: ESConnection, vector_dims: int):
        """
        Initialize the Elasticsearch Document Manager.

        :param es_connection: The Elasticsearch connection instance.
        :param vector_dims: The dimensions of the vector embeddings.
        """
        self.es = es_connection.client
        self.vector_dims = vector_dims
        self.logger = logger

    @ensure_connection
    def insert_document(self, 
                        index_name: str, 
                        question: str, 
                        question_keywords: List[str], 
                        question_vector: List[float], 
                        answer: str, 
                        answer_keywords: List[str], 
                        answer_vector: List[float]) -> Dict[str, Any]:
        """
        Insert document into Elasticsearch according to the provided mapping schema.

        :param index_name: The name of the index.
        :param question: The question text.
        :param question_keywords: The keywords extracted from the question.
        :param question_vector: The vector embedding of the question.
        :param answer: The answer text.
        :param answer_keywords: The keywords extracted from the answer.
        :param answer_vector: The vector embedding of the answer.
        :return: The result of the document insertion operation.
        """
        
        # Check if the lengths of the vectors match the expected dimensions
        if len(question_vector) != self.vector_dims:
            raise ValueError(f"Expected vector of length {self.vector_dims}, but got {len(question_vector)}")
        if len(answer_vector) != self.vector_dims:
            raise ValueError(f"Expected vector of length {self.vector_dims}, but got {len(answer_vector)}")
        
        # Construct the document body
        body = {
            "Question": question,
            "Question_Keywords": question_keywords,
            "Question_Vector": question_vector,
            "Answer": answer,
            "Answer_Keywords": answer_keywords,
            "Answer_Vector": answer_vector
        }
        
        # Attempt to insert the document
        try:
            res = self.es.index(index=index_name, body=body)
            return res
        except Exception as e:
            self.logger.error(f"Error inserting document: {e}")
            raise



# Search
class ESSearch:
    """
    This class is responsible for performing search operations in Elasticsearch.
    It supports three types of search: semantic, full-text, and hybrid.
    """

    # Define valid search scopes and types
    VALID_SEARCH_SCOPES = {'Questions', 'Answers', 'Q&A'}
    VALID_SEARCH_TYPES = {'semantic', 'full_text', 'hybrid'}

    def __init__(self, es_connection: ESConnection, vector_dims: int):
        """
        Initialize the Elasticsearch Search instance.

        :param es_connection: An instance of the Elasticsearch connection.
        :param vector_dims: The dimensions of the vector embeddings.
        """
        self.es = es_connection.client
        self.vector_dims = vector_dims
        self.logger = logger

    @ensure_connection
    def hybrid_search(self, 
                      index_name: str, 
                      input_question: str = None, 
                      input_question_keywords: List[str] = None, 
                      input_question_vector: List[float] = None, 
                      k: int = 5, 
                      num_candidates: int = 10,
                      search_scope: str = 'Q&A',
                      search_type: str = 'hybrid') -> Dict[str, Any]:
        """
        Execute a hybrid search on Elasticsearch.

        :param index_name: The name of the index to search.
        :param input_question: The question to search for.
        :param input_question_keywords: The keywords extracted from the question.
        :param input_question_vector: The vector embedding of the question.
        :param k: The number of results to return.
        :param num_candidates: The number of candidates to consider during the search.
        :param search_scope: The scope of the search. Can be 'Questions', 'Answers', or 'Q&A'.
        :param search_type: The type of search to perform. Can be 'semantic', 'full_text', or 'hybrid'.
        :return: The result of the search operation.
        """
        # Construct the search body
        search_body = self._construct_search_body(
            input_question, 
            input_question_keywords, 
            input_question_vector, 
            search_scope, 
            search_type, 
            k, 
            num_candidates
        )

        # Execute the search and return the result
        return self._execute_search(index_name, search_body)

    def _validate_search_parameters(self, search_scope: str, search_type: str):
        """
        Validate the search parameters.

        :param search_scope: The scope of the search. Must be one of the valid search scopes.
        :param search_type: The type of search to perform. Must be one of the valid search types.
        :raises ValueError: If the search scope or type is not valid.
        """
        if search_scope not in self.VALID_SEARCH_SCOPES:
            raise ValueError(f"Invalid search_scope. Acceptable values are {', '.join(self.VALID_SEARCH_SCOPES)}")
        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(f"Invalid search_type. Acceptable values are {', '.join(self.VALID_SEARCH_TYPES)}")

    def _construct_search_body(self, input_question, input_question_keywords, input_question_vector, search_scope, search_type, k, num_candidates):
        """
        Construct the search body for the Elasticsearch query.

        :param input_question: The question to search for.
        :param input_question_keywords: The keywords extracted from the question.
        :param input_question_vector: The vector embedding of the question.
        :param search_scope: The scope of the search.
        :param search_type: The type of search to perform.
        :param k: The number of results to return.
        :param num_candidates: The number of candidates to consider during the search.
        :return: The search body for the Elasticsearch query.
        """
        # Initialize the search body
        search_body = {"size": k}
        should_clauses = []

        # Construct the full-text or hybrid search query
        if search_type in ['full_text', 'hybrid']:
            if input_question:
                text_query = self._text_query(input_question, search_scope)
                text_query["bool"]["boost"] = 0.4 if search_type == 'full_text' else 0.1
                should_clauses.append(text_query)

            if input_question_keywords:
                keyword_query = self._keyword_query(input_question_keywords, search_scope)
                keyword_query["bool"]["boost"] = 0.6 if search_type == 'full_text' else 0.5
                should_clauses.append(keyword_query)

        # Construct the semantic or hybrid search query
        if search_type in ['semantic', 'hybrid']:
            if input_question_vector:
                knn_query = self._knn_query(input_question_vector, search_scope, k, num_candidates)
                knn_query["boost"] = 1.0 if search_type == 'semantic' else 0.4
                # Add the knn query separately
                search_body["knn"] = [knn_query]

        # Add the constructed queries to the search body
        search_body["query"] = {"bool": {"should": should_clauses}}

        return search_body

    def _text_query(self, input_question, search_scope):
        """
        Construct the text query for the Elasticsearch query.

        :param input_question: The question to search for.
        :param search_scope: The scope of the search.
        :return: The text query for the Elasticsearch query.
        """
        should_clauses = []
        if search_scope in ['Questions', 'Q&A']:
            should_clauses.append({"match": {"Question": input_question}})
        if search_scope in ['Answers', 'Q&A']:
            should_clauses.append({"match": {"Answer": input_question}})
        return {"bool": {"should": should_clauses}}

    def _keyword_query(self, input_question_keywords, search_scope):
        """
        Construct the keyword query for the Elasticsearch query.

        :param input_question_keywords: The keywords extracted from the question.
        :param search_scope: The scope of the search.
        :return: The keyword query for the Elasticsearch query.
        """
        should_clauses = []
        if search_scope in ['Questions', 'Q&A']:
            should_clauses.append({"terms": {"Question_Keywords": input_question_keywords}})
        if search_scope in ['Answers', 'Q&A']:
            should_clauses.append({"terms": {"Answer_Keywords": input_question_keywords}})
        return {"bool": {"should": should_clauses}}

    def _knn_query(self, input_question_vector, search_scope, k, num_candidates):
        """
        Construct the k-nearest neighbors (knn) query for the Elasticsearch query.

        :param input_question_vector: The vector embedding of the question.
        :param search_scope: The scope of the search.
        :param k: The number of results to return.
        :param num_candidates: The number of candidates to consider during the search.
        :return: The knn query for the Elasticsearch query.
        """
        field = "Question_Vector" if search_scope in ['Questions', 'Q&A'] else "Answer_Vector"
        # Default behavior for non-exact knn_type
        return {
            "field": field,
            "query_vector": input_question_vector,
            "k": k,
            "num_candidates": num_candidates
        }

    def _execute_search(self, index_name, search_body):
        """
        Execute the search operation in Elasticsearch.

        :param index_name: The name of the index to search.
        :param search_body: The search body for the Elasticsearch query.
        :return: The result of the search operation.
        :raises Exception: If an error occurs during the search operation.
        """
        try:
            return self.es.search(index=index_name, body=search_body)
        except Exception as e:
            self.logger.error(f"Error executing hybrid search: {e}")
            raise


class VertexLLMCacheConfig:
    """
    This class is responsible for storing the configuration parameters for the VertexLLMCache.
    """
    def __init__(self, host: str, port: int, username: str, password: str, index_name: str, cert_path: str,
                 model_name: str = 'text-bison', vector_dims: int = 768, similarity_type: str = "dot_product",
                 knn_type: str = "exact"):
        """
        Initialize the VertexLLMCacheConfig instance.

        :param host: The host IP address for the Elasticsearch server.
        :param port: The port number for the Elasticsearch server.
        :param username: The username for the Elasticsearch server.
        :param password: The password for the Elasticsearch server.
        :param index_name: The name of the index in Elasticsearch.
        :param cert_path: The path to the certificate for the Elasticsearch server.
        :param model_name: The name of the model to use for embeddings. Default is 'text-bison'.
        :param vector_dims: The dimensions of the vector embeddings. Default is 768.
        :param similarity_type: The type of similarity to use for the embeddings. Default is 'dot_product'.
        :param knn_type: The type of k-nearest neighbors algorithm to use. Default is 'exact'.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index_name = index_name
        self.cert_path = cert_path
        self.model_name = model_name
        self.vector_dims = vector_dims
        self.similarity_type = similarity_type
        self.knn_type = knn_type


class VertexLLMCache:
    """
    This class is responsible for managing a cache of questions and answers using VertexAI and Elasticsearch.
    It supports inserting data into the cache and searching for data in the cache.
    """

    # Define the template for extracting entities from text
    TEMPLATE = """
    Extract Entities from Text Separated by Commas
    Text: {text}
    Output:
    """

    def __init__(self, config: VertexLLMCacheConfig, llm: VertexAI = None, embeddings: VertexAIEmbeddings = None):
        """
        Initialize the VertexLLMCache instance.

        :param config: An instance of the VertexLLMCacheConfig.
        :param llm: An instance of the VertexAI. If not provided, a new instance will be created.
        :param embeddings: An instance of the VertexAIEmbeddings. If not provided, a new instance will be created.
        """
        self.llm = llm or VertexAI(model_name=config.model_name)
        self.embeddings = embeddings or VertexAIEmbeddings()
        self.prompt = PromptTemplate(template=self.TEMPLATE, input_variables=["text"])
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

        # Establish connection to Elasticsearch
        self.vertex_cache_client = ESCacheClient(
            host=config.host, port=config.port, vector_dims=config.vector_dims,
            username=config.username, password=config.password, cert_path=config.cert_path
        )
        self.vertex_cache_client.setup_index(
            index_name=config.index_name, force_deletion=True,
            vector_dims=config.vector_dims, similarity_type=config.similarity_type
        )
        self.index_name = config.index_name

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the given text.

        :param text: The text to extract keywords from.
        :return: A list of keywords extracted from the text.
        """
        keywords_str = self.llm_chain.run({"text": text})
        return keywords_str.split(",")

    def insert_into_cache(self, question: str, answer: str) -> None:
        """
        Insert a question and its corresponding answer into the cache.

        :param question: The question to insert into the cache.
        :param answer: The answer to the question.
        """
        # Extract keywords and embeddings from the question and answer
        qn_keywords = self._extract_keywords(question)
        ans_keywords = self._extract_keywords(answer)
        question_embedding = self.embeddings.embed_query(question)
        answer_embedding = self.embeddings.embed_query(answer)

        # Insert the question and answer into the cache
        self.vertex_cache_client.insert_document(
            index_name=self.index_name,
            question=question,
            question_keywords=qn_keywords,
            question_vector=question_embedding,
            answer=answer,
            answer_keywords=ans_keywords,
            answer_vector=answer_embedding
        )

    def _search_from_cache(self, input_qn: str, search_scope: str, search_type: str) -> Union[Dict, List]:
        """
        Search for a question in the cache.

        :param input_qn: The question to search for.
        :param search_scope: The scope of the search.
        :param search_type: The type of search to perform.
        :return: The result of the search operation.
        """
        # Extract keywords and embeddings from the input question
        input_qn_keywords = self._extract_keywords(input_qn)
        # Perform the search and return the result
        return self.vertex_cache_client.hybrid_search(
            index_name=self.index_name,
            input_question=input_qn,
            input_question_vector=self.embeddings.embed_query(input_qn),
            input_question_keywords=input_qn_keywords,
            search_scope=search_scope,
            search_type=search_type
        )

    def l1_search(self, input_qn:str, search_type:str = "hybrid") -> Union[Dict, List]:
        """
        Perform a level 1 search in the cache.
        This search only considers the questions in the cache.

        :param input_qn: The question to search for.
        :param search_type: The type of search to perform. Default is 'hybrid'.
        :return: The result of the search operation.
        """
        return self._search_from_cache(input_qn=input_qn, search_scope="Questions", search_type=search_type)
    
    def l2_search(self, input_qn:str, search_type:str = "hybrid") -> Union[Dict, List]:
        """
        Perform a level 2 search in the cache.
        This search only considers the answers in the cache.

        :param input_qn: The question to search for.
        :param search_type: The type of search to perform. Default is 'hybrid'.
        :return: The result of the search operation.
        """
        return self._search_from_cache(input_qn=input_qn, search_scope="Answers", search_type=search_type)
    
    def l3_search(self, input_qn:str, search_type:str = "hybrid") -> Union[Dict, List]:
        return self._search_from_cache(input_qn=input_qn, search_scope="Q&A", search_type=search_type)




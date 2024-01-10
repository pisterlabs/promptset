from weaviate import Client, AuthApiKey
from dataclasses import dataclass
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Union, Callable
from torch import cuda
from tqdm import tqdm
import time

class WeaviateClient(Client):
    '''
    A python native Weaviate Client class that encapsulates Weaviate functionalities 
    in one object. Several convenience methods are added for ease of use.

    Args
    ----
    api_key: str
        The API key for the Weaviate Cloud Service (WCS) instance.
        https://console.weaviate.cloud/dashboard

    endpoint: str
        The url endpoint for the Weaviate Cloud Service instance.

    model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2'
        The name or path of the SentenceTransformer model to use for vector search.
        Will also support OpenAI text-embedding-ada-002 model.  This param enables 
        the use of most leading models on MTEB Leaderboard: 
        https://huggingface.co/spaces/mteb/leaderboard
    openai_api_key: str=None
        The API key for the OpenAI API. Only required if using OpenAI text-embedding-ada-002 model.
    '''    
    def __init__(self, 
                 api_key: str,
                 endpoint: str,
                 model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2',
                 openai_api_key: str=None,
                 **kwargs
                ):
        auth_config = AuthApiKey(api_key=api_key)
        super().__init__(auth_client_secret=auth_config,
                         url=endpoint,
                         **kwargs)    
        self.model_name_or_path = model_name_or_path
        self.openai_model = False
        if self.model_name_or_path == 'text-embedding-ada-002':
            if not openai_api_key:
                raise ValueError(f'OpenAI API key must be provided to use this model: {self.model_name_or_path}')
            self.model = OpenAI(api_key=openai_api_key)
            self.openai_model = True
        else: 
            self.model = SentenceTransformer(self.model_name_or_path) if self.model_name_or_path else None

        self.display_properties = ['title', 'video_id', 'length', 'thumbnail_url', 'views', 'episode_url', \
                                    'doc_id', 'guest', 'content']  # 'playlist_id', 'channel_id', 'author'
        
    def show_classes(self) -> Union[List[dict], str]:
        '''
        Shows all available classes (indexes) on the Weaviate instance.
        '''
        classes = self.cluster.get_nodes_status()[0]['shards']
        if classes:
            return [d['class'] for d in classes]
        else: 
            return "No classes found on cluster."

    def show_class_info(self) -> Union[List[dict], str]:
        '''
        Shows all information related to the classes (indexes) on the Weaviate instance.
        '''
        classes = self.cluster.get_nodes_status()[0]['shards']
        if classes:
            return [d for d in classes]
        else: 
            return "No classes found on cluster."

    def show_class_properties(self, class_name: str) -> Union[dict, str]:
        '''
        Shows all properties of a class (index) on the Weaviate instance.
        '''
        classes = self.schema.get()
        if classes:
            all_classes = classes['classes']
            for d in all_classes:
                if d['class'] == class_name:
                    return d['properties']
            return f'Class "{class_name}" not found on host'
        return f'No Classes found on host'
    
    def show_class_config(self, class_name: str) -> Union[dict, str]:
        '''
        Shows all configuration of a class (index) on the Weaviate instance.
        '''
        classes = self.schema.get()
        if classes:
            all_classes = classes['classes']
            for d in all_classes:
                if d['class'] == class_name:
                    return d
            return f'Class "{class_name}" not found on host'
        return f'No Classes found on host'
    
    def delete_class(self, class_name: str) -> str:
        '''
        Deletes a class (index) on the Weaviate instance, if it exists.
        '''
        available = self._check_class_avialability(class_name)
        if isinstance(available, bool):
            if available:
                self.schema.delete_class(class_name)
                not_deleted = self._check_class_avialability(class_name)
                if isinstance(not_deleted, bool):
                    if not_deleted:
                        return f'Class "{class_name}" was not deleted. Try again.'
                    else: 
                        return f'Class "{class_name}" deleted'
                return f'Class "{class_name}" deleted and there are no longer any classes on host'
            return f'Class "{class_name}" not found on host'
        return available
    
    def _check_class_avialability(self, class_name: str) -> Union[bool, str]:
        '''
        Checks if a class (index) exists on the Weaviate instance.
        '''
        classes = self.schema.get()
        if classes:
            all_classes = classes['classes']
            for d in all_classes:
                if d['class'] == class_name:
                    return True
            return False
        else: 
            return f'No Classes found on host'
        
    def format_response(self, 
                         response: dict,
                         class_name: str
                         ) -> List[dict]:
        '''
        Formats json response from Weaviate into a list of dictionaries.
        Expands _additional fields if present into top-level dictionary.
        '''
        if response.get('errors'):
            return response['errors'][0]['message']
        results = []
        hits = response['data']['Get'][class_name]
        for d in hits:
            temp = {k:v for k,v in d.items() if k != '_additional'}
            if d.get('_additional'):
                for key in d['_additional']:
                    temp[key] = d['_additional'][key]
            results.append(temp)
        return results
        
    def keyword_search(self,
                       request: str,
                       class_name: str,
                       properties: List[str]=['content'],
                       limit: int=10,
                       where_filter: dict=None,
                       display_properties: List[str]=None,
                       return_raw: bool=False) -> Union[dict, List[dict]]:
        '''
        Executes Keyword (BM25) search. 

        Args
        ----
        query: str
            User query.
        class_name: str
            Class (index) to search.
        properties: List[str]
            List of properties to search across.
        limit: int=10
            Number of results to return.
        display_properties: List[str]=None
            List of properties to return in response.
            If None, returns all properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        display_properties = display_properties if display_properties else self.display_properties
        response = (self.query
                    .get(class_name, display_properties)
                    .with_bm25(query=request, properties=properties)
                    .with_additional(['score', "id"])
                    .with_limit(limit)
                    )
        response = response.with_where(where_filter).do() if where_filter else response.do()
        if return_raw:
            return response
        else: 
            return self.format_response(response, class_name)

    def vector_search(self,
                      request: str,
                      class_name: str,
                      limit: int=10,
                      where_filter: dict=None,
                      display_properties: List[str]=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                      ) -> Union[dict, List[dict]]:
        '''
        Executes vector search using embedding model defined on instantiation 
        of WeaviateClient instance.
        
        Args
        ----
        query: str
            User query.
        class_name: str
            Class (index) to search.
        limit: int=10
            Number of results to return.
        display_properties: List[str]=None
            List of properties to return in response.
            If None, returns all properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        display_properties = display_properties if display_properties else self.display_properties
        query_vector = self._create_query_vector(request, device=device)
        response = (
                    self.query
                    .get(class_name, display_properties)
                    .with_near_vector({"vector": query_vector})
                    .with_limit(limit)
                    .with_additional(['distance'])
                    )
        response = response.with_where(where_filter).do() if where_filter else response.do()
        if return_raw:
            return response
        else: 
            return self.format_response(response, class_name)     
    
    def _create_query_vector(self, query: str, device: str) -> List[float]:
        '''
        Creates embedding vector from text query.
        '''
        return self.get_openai_embedding(query) if self.openai_model else self.model.encode(query, device=device).tolist()
    
    def get_openai_embedding(self, query: str) -> List[float]:
        '''
        Gets embedding from OpenAI API for query.
        '''
        embedding = self.model.embeddings.create(input=query, model='text-embedding-ada-002').model_dump()
        if embedding:
            return embedding['data'][0]['embedding']
        else:
           raise ValueError(f'No embedding found for query: {query}')
        
    def hybrid_search(self,
                      request: str,
                      class_name: str,
                      properties: List[str]=['content'],
                      alpha: float=0.5,
                      limit: int=10,
                      where_filter: dict=None,
                      display_properties: List[str]=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                     ) -> Union[dict, List[dict]]:
        '''
        Executes Hybrid (BM25 + Vector) search.
        
        Args
        ----
        query: str
            User query.
        class_name: str
            Class (index) to search.
        properties: List[str]
            List of properties to search across (using BM25)
        alpha: float=0.5
            Weighting factor for BM25 and Vector search.
            alpha can be any number from 0 to 1, defaulting to 0.5:
                alpha = 0 executes a pure keyword search method (BM25)
                alpha = 0.5 weighs the BM25 and vector methods evenly
                alpha = 1 executes a pure vector search method
        limit: int=10
            Number of results to return.
        display_properties: List[str]=None
            List of properties to return in response.
            If None, returns all properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        display_properties = display_properties if display_properties else self.display_properties
        query_vector = self._create_query_vector(request, device=device)
        response = (
                    self.query
                    .get(class_name, display_properties)
                    .with_hybrid(query=request,
                                 alpha=alpha,
                                 vector=query_vector,
                                 properties=properties,
                                 fusion_type='relativeScoreFusion') #hard coded option for now
                    .with_additional(["score", "explainScore"])
                    .with_limit(limit)
                    )
        
        response = response.with_where(where_filter).do() if where_filter else response.do()
        if return_raw:
            return response
        else: 
            return self.format_response(response, class_name)
        
        
class WeaviateIndexer:

    def __init__(self,
                 client: WeaviateClient,
                 batch_size: int=150,
                 num_workers: int=4,
                 dynamic: bool=True,
                 creation_time: int=5,
                 timeout_retries: int=3,
                 connection_error_retries: int=3,
                 callback: Callable=None,
                 ):
        '''
        Class designed to batch index documents into Weaviate. Instantiating
        this class will automatically configure the Weaviate batch client.
        '''
        self._client = client
        self._callback = callback if callback else self._default_callback
        
        self._client.batch.configure(batch_size=batch_size,
                                     num_workers=num_workers,
                                     dynamic=dynamic,
                                     creation_time=creation_time,
                                     timeout_retries=timeout_retries,
                                     connection_error_retries=connection_error_retries,
                                     callback=self._callback
                                    )
        
    def _default_callback(self, results: dict):
        """
        Check batch results for errors.

        Parameters
        ----------
        results : dict
            The Weaviate batch creation return value.
        """

        if results is not None:
            for result in results:
                if "result" in result and "errors" in result["result"]:
                    if "error" in result["result"]["errors"]:
                        print(result["result"])

    def batch_index_data(self,
                         data: List[dict], 
                         class_name: str,
                         vector_property: str='content_embedding'
                         ) -> None:
        '''
        Batch function for fast indexing of data onto Weaviate cluster. 
        This method assumes that self._client.batch is already configured.
        '''
        start = time.perf_counter()
        with self._client.batch as batch:
            for d in tqdm(data):
                
                #define single document 
                properties = {k:v for k,v in d.items() if k != vector_property}
                try:
                    #add data object to batch
                    batch.add_data_object(
                                        data_object=properties,
                                        class_name=class_name,
                                        vector=d[vector_property]
                                        )
                except Exception as e:
                    print(e)
                    continue

        end = time.perf_counter() - start
    
        print(f'Batch job completed in {round(end/60, 2)} minutes.')
        class_info = self._client.show_class_info()
        for i, c in enumerate(class_info):
            if c['class'] == class_name:
                print(class_info[i])
        self._client.batch.shutdown()

@dataclass
class WhereFilter:

    '''
    Simplified interface for constructing a WhereFilter object.

    Args
    ----
    path: List[str]
        List of properties to filter on.
    operator: str
        Operator to use for filtering. Options: ['And', 'Or', 'Equal', 'NotEqual', 
        'GreaterThan', 'GreaterThanEqual', 'LessThan', 'LessThanEqual', 'Like', 
        'WithinGeoRange', 'IsNull', 'ContainsAny', 'ContainsAll']
    value[dataType]: Union[int, bool, str, float, datetime]
        Value to filter on. The dataType suffix must match the data type of the 
        property being filtered on. At least and only one value type must be provided. 
    '''
    path: List[str]
    operator: str
    valueInt: int=None
    valueBoolean: bool=None
    valueText: str=None
    valueNumber: float=None
    valueDate = None

    def post_init(self):
        operators = ['And', 'Or', 'Equal', 'NotEqual','GreaterThan', 'GreaterThanEqual', 'LessThan',\
                      'LessThanEqual', 'Like', 'WithinGeoRange', 'IsNull', 'ContainsAny', 'ContainsAll']
        if self.operator not in operators:
            raise ValueError(f'operator must be one of: {operators}, got {self.operator}')
        values = [self.valueInt, self.valueBoolean, self.valueText, self.valueNumber, self.valueDate]
        if not any(values):
            raise ValueError('At least one value must be provided.')
        if len(values) > 1:
            raise ValueError('At most one value can be provided.')
    
    def todict(self):
        return {k:v for k,v in self.__dict__.items() if v is not None}
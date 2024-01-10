from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast

from langchain.schema import BaseStore
from langchain.utilities.redis import get_client
from langchain.schema import Document

class RedisJSONStore(BaseStore[str, Document]):
    """BaseStore implementation using RedisJSON as the underlying store.
    Overcomes limitations of RedisStore, that can only store scalar values, no Document objects.

    LIMITATION - Only works for storing langchain.schema.Document objects.

    Examples:
        Create a RedisStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the RedisStore with a Redis connection
            from langchain.storage import RedisStore
            from langchain.utilities.redis import get_client

            client = get_client('redis://localhost:6379')
            redis_store = RedisJSONStore(client)

            Document doc1 = Document(page_content="foo", metadata={"foo": "bar"})	
            Document doc2 = Document(page_content="bar", metadata={"bar": "foo"})

            # Set values for keys
            redis_store.mset([("key1", doc1), ("key2", doc2)])

            # Get values for keys
            values = redis_store.mget(["key1", "key2"])
            for doc in values:
                print(doc.page_content)
                print(doc.metadata)

            # Delete keys
            redis_store.mdelete(["key1"])

            # Iterate over keys
            for key in redis_store.yield_keys():
                print(key)
    """

    def __init__(
        self,
        *,
        client: Any = None,
        redis_url: Optional[str] = None,
        client_kwargs: Optional[dict] = None,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Initialize the RedisStore with a Redis connection.

        Must provide either a Redis client or a redis_url with optional client_kwargs.

        Args:
            client: A Redis connection instance
            redis_url: redis url
            client_kwargs: Keyword arguments to pass to the Redis client
            ttl: time to expire keys in seconds if provided,
                 if None keys will never expire
            namespace: if provided, all keys will be prefixed with this namespace
        """
        try:
            from redis import Redis
        except ImportError as e:
            raise ImportError(
                "The RedisStore requires the redis library to be installed. "
                "pip install redis"
            ) from e

        if client and redis_url or client and client_kwargs:
            raise ValueError(
                "Either a Redis client or a redis_url with optional client_kwargs "
                "must be provided, but not both."
            )

        if client:
            if not isinstance(client, Redis):
                raise TypeError(
                    f"Expected Redis client, got {type(client).__name__} instead."
                )
            _client = client
        else:
            if not redis_url:
                raise ValueError(
                    "Either a Redis client or a redis_url must be provided."
                )
            _client = get_client(redis_url, **(client_kwargs or {}))

        self.client = _client

        if not isinstance(ttl, int) and ttl is not None:
            raise TypeError(f"Expected int or None, got {type(ttl)} instead.")

        self.ttl = ttl
        self.namespace = namespace


    def _get_prefixed_key(self, key: str) -> str:
        """Get the key with the namespace prefix.

        Args:
            key (str): The original key.

        Returns:
            str: The key with the namespace prefix.
        """
        delimiter = ":"
        if self.namespace:
            return f"{self.namespace}{delimiter}{key}"
        return key

    #COMO ERA NO REDIS STORE, APAGAR
    # def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
    #     """Get the values associated with the given keys."""
    #     return cast(
    #         List[Optional[bytes]],
    #         self.client.mget([self._get_prefixed_key(key) for key in keys]),
    #     )
    
    #FEITO BASEADO NO INMEMORY STORE
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        json_records = self.client.json().mget(keys=[self._get_prefixed_key(key) for key in keys], path=".")

        document_list = [self.__dict_to_document(json_record) for json_record in json_records]

        result = [doc for doc in document_list]

        return result

 #COMO ERA NO REDIS STORE, APAGAR
    #  def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
    #     """Set the given key-value pairs."""
    #     pipe = self.client.pipeline()

    #     for key, value in key_value_pairs:
    #         pipe.set(self._get_prefixed_key(key), value, ex=self.ttl)
    #     pipe.execute()

    #Serialização de Document de e para json
    #------------------------------------------------
    def __document_to_dict(self, document: Document) -> dict:
        return {"page_content": document.page_content, "metadata": document.metadata}
    
    def __dict_to_document(self, dict: dict) -> Document:
        return Document(page_content=dict['page_content'], metadata=dict['metadata'])
    
    #FEITO BASEADO NO INMEMORY STORE
    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        #Gera uma lista de tuples contendo: chave, jsonpath (que é . para representar o objeto inteiro), e um dictionary com as propriedades do Document, que representa o objeto json em si
        json_tuple_list = [
            (self._get_prefixed_key(key), ".", self.__document_to_dict(doc)) for key, doc in key_value_pairs
        ]

        self.client.json().mset(json_tuple_list)
        



    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        _keys = [self._get_prefixed_key(key) for key in keys]
        self.client.delete(*_keys)


    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store."""
        if prefix:
            pattern = self._get_prefixed_key(prefix)
        else:
            pattern = self._get_prefixed_key("*")
        scan_iter = cast(Iterator[bytes], self.client.scan_iter(match=pattern))
        for key in scan_iter:
            decoded_key = key.decode("utf-8")
            if self.namespace:
                relative_key = decoded_key[len(self.namespace) + 1 :]
                yield relative_key
            else:
                yield decoded_key

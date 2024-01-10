from ..config import *
from langchain.vectorstores import VectorStore
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List,Dict, Tuple
import logging
from enum import Enum
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from langchain.storage import LocalFileStore, RedisStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
import pprint
from sqlalchemy.sql import text

class OrderBy(Enum):
    RATING = "rating_value"
    POPULARITY = "rating_count"

class Order(Enum):
    ASC = "asc"
    DESC = "desc"

class FilterType(Enum):
    ACTOR = "actor"
    DIRECTOR = "director"
    GENRE = "genre"
    RATING = "rating"

class Filter(ABC):
    def __init__(self, filter_type: FilterType, *args, **kwargs):
        self.filter_type = filter_type

    @abstractmethod
    def __repr__(self):
        pass

class RatingFilter(Filter):
    def __init__(self, rating: float
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.RATING, *args, **kwargs)
        self.rating = rating

    def __repr__(self):
        return f"rating_value >= {self.rating}"

class ActorFilter(Filter):
    def __init__(self, actor: str
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.ACTOR, *args, **kwargs)
        self.actor = actor

    def __repr__(self):
        return f"actors LIKE '%{self.actor}%'"
    
class DirectorFilter(Filter):
    def __init__(self, director: str
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.DIRECTOR, *args, **kwargs)
        self.director = director

    def __repr__(self):
        return f"director LIKE '%{self.director}%'"
    
class GenreFilter(Filter):
    def __init__(self, genre: str
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.GENRE, *args, **kwargs)
        self.genre = genre

    def __repr__(self):
        return f"genre LIKE '%{self.genre}%'"

class SearchEngine():
    """
    """
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        vectorstore: Redis,
        session: Session,
        max_documents: int = 100,
        similarity_thresh: float = SIMILARITY_THRESH
    ) -> None:
        """
        Arguments:
            embeddings: OpenAIEmbeddings object
            vectorstore: VectorStore object
            max_documents: Maximum number of documents to search
            similarity_thresh: Threshold for similarity
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.session = session
        self.max_documents = max_documents
        self.similarity_thresh = similarity_thresh

    def _construct_search_query_with_uuids(
        self, 
        uuids: List[str],
        order_by: str,
        order: str,
        limit: int,
        filters: List[Filter],
    ) -> str:
        uuids_str = ""
        for uuid in uuids:
            uuids_str += f"'{uuid}',"
        uuids_str = uuids_str[:-1]
        uuids_str = "(" + uuids_str + ")"
        suffix = f"ORDER BY {order_by} {order} LIMIT {limit}"
        query = f"""
SELECT * FROM Movie WHERE id IN {uuids_str}"""
        for filter in filters:
            query += f"\n   AND {filter}"
        query += f" {suffix}"
        query += ";"
        return query
    
    def _construct_search_query(
        self, 
        uuids: List[str],
        order_by: str,
        order: str,
        limit: int,
        filters: List[Filter],
    ) -> str:
        if len(uuids) != 0:
            return self._construct_search_query_with_uuids(
                uuids, order_by, order, limit, filters)
        suffix = f"ORDER BY {order_by} {order} LIMIT {limit}"
        query = f"""
SELECT * FROM Movie"""
        isFirstFlag = True
        for filter in filters:
            if isFirstFlag:
                query += f"\nWHERE {filter}"
                isFirstFlag = False
            else:
                query += f"\n   AND {filter}"
        query += f" {suffix}"
        query += ";"
        return query
    
    
    def search(
        self,
        search_term: str,
        order_by: OrderBy = OrderBy.RATING,
        order: Order = Order.DESC,
        limit: int = MAX_DOCUMENTS,
        filters: List[Filter] = []
    ) -> List[Dict]:
        """
        Arguments:
            search_term: Search term
            order_by: Order by which to sort the results
            order: Order of the results
            limit: Maximum number of results to return
            filters: List of filters to apply to the results

        """
        # logging.debug("Searching in vectorstore")
        uuids = []
        if search_term != "":
            retriever = self.vectorstore.as_retriever(search_type="similarity")
            retriever.k = self.max_documents
            retriever.score_threshold = self.similarity_thresh
            results = retriever.get_relevant_documents(search_term)
            uuids = [result.metadata["uuid"] for result in results]
        logging.debug("Search in vectorstore complete")
        
        query = self._construct_search_query(
            uuids=uuids,
            order_by=order_by.value,
            order=order.value,
            limit=limit,
            filters=filters,
        )
        print(f"Search Query: {query}")
        # exit()
        try:

            resp = self.session.execute(text(query))
            movies: List = resp.fetchall()
        except Exception as e:
            # logging.error(f"Error while executing query {query}: {e}")
            raise e
 
        movieDicts = []
        for movie in movies:
            movieDicts.append({
                "name": movie.name,
                "url": movie.url,
                "poster": movie.poster,
                "rating": movie.rating_value,
            })
        # logging.debug(f"Search results: {movieDicts}")
        return movieDicts


def test(prompt):
    from ..models import session
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                             redis_url=REDIS_URL,
                                             embedding=embeddings)
    se = SearchEngine(embeddings=embeddings,
                      vectorstore=vectorstore,
                      session=session,
                      max_documents=10,similarity_thresh=0.2)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(se.search(prompt, 
                    order_by=OrderBy.RATING,
                    order=Order.DESC,
                    limit=10,
                    filters=[]))

if __name__ == "__main__":
    while True:
        test(input("Prompt: "))
        print("----------------")
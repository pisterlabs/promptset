from .search_engine import (
    SearchEngine, 
    OrderBy,
    Order,
    RatingFilter,
    ActorFilter,
    DirectorFilter,
    GenreFilter,
)
from ..config import *
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessage, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import List,Dict
from ..models import session

SEARCH_ENGINE_ARGS_SCHEMA = {
    "title": "SearchEngineArgs",
    "description": "Identify the arguments for the search engine",
    "type": "object",
    "properties": {
        "search_term": {"title": "search_term", "description": "Description about the movie that can contain the movie name, plotline or any other thing that hints towards a movie", "type": "string"},
        "order_by": {"title": "order_by", "description": "Order the results by", "type": "string", "enum": ["rating", "popularity"]},
        "is_descending": {"title": "is_descending", "description": "Order in descending order or not", "type": "boolean"},
        "rating_filter": {"title": "rating_filter", "description": "Filter by rating", "type": "integer","minimum": 0, "maximum": 10},
        "actor_filter": {"title": "actor_filter", "description": "List of Actors starring in the movie", "type": "array", "items": {"type": "string"}},
        "director_filter": {"title": "director_filter", "description": "List of Directors who directed the movie", "type": "array", "items": {"type": "string"}},
        "genre_filter": {"title": "genre_filter", "description": "List of genres that the movie is of", "type": "array", "items": {"type": "string"}},
    },
    "required": [],
}


class SearchEngineWithLLM():
    """
    A wrapper around the search engine that uses a language model to extract the arguments for the search engine
    """
    def __init__(
        self, 
        base_llm: BaseLLM,
        search_engine: SearchEngine,
        *args, 
        **kwargs
    ) -> None:
        """
        Args:
            base_llm (BaseLLM): The language model to use for extracting arguments
            search_engine (SearchEngine): The search engine to use for searching
        """
        self.search_engine = search_engine
        self.base_llm = base_llm
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting information in structured formats to be used to search for movies."
            ),
            HumanMessage(content="Use the given format to extract information from the following input:"),
            HumanMessagePromptTemplate.from_template("Input: {input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)
        self.llm_chain = create_structured_output_chain(SEARCH_ENGINE_ARGS_SCHEMA, base_llm, prompt, verbose=True)

    def construct_se_kwargs(self, kwargs: Dict) -> Dict:
        se_kwargs = {"filters": []}
        se_kwargs["search_term"] = kwargs.get("search_term", "")
        if kwargs.get("order_by"):
            if kwargs["order_by"] == "rating":
                se_kwargs["order_by"] = OrderBy.RATING
            elif kwargs["order_by"] == "popularity":
                se_kwargs["order_by"] = OrderBy.POPULARITY
        if kwargs.get("is_descending"):
            se_kwargs["order"] = Order.DESC if kwargs["is_descending"] else Order.ASC
        if kwargs.get("rating_filter"):
            se_kwargs["filters"].append(RatingFilter(kwargs["rating_filter"]))            
        if kwargs.get("actor_filter"):
            for actor in kwargs["actor_filter"]:
                se_kwargs["filters"].append(ActorFilter(actor))
        if kwargs.get("director_filter"):
            for director in kwargs["director_filter"]:
                se_kwargs["filters"].append(DirectorFilter(director))
        if kwargs.get("genre_filter"):
            for genre in kwargs["genre_filter"]:
                se_kwargs["filters"].append(GenreFilter(genre))
        print(se_kwargs)
        return se_kwargs
    
    def search(self,prompt) -> List:
        kwargs = self.llm_chain.run(input=prompt)
        se_kwargs = self.construct_se_kwargs(kwargs)
        return self.search_engine.search(**se_kwargs)


def build_search_engine():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                             redis_url=REDIS_URL,
                                             embedding=embeddings)
    se = SearchEngine(
        embeddings=embeddings,
        vectorstore=vectorstore,
        session=session,
        max_documents=MAX_DOCUMENTS,
        similarity_thresh=SIMILARITY_THRESH,
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0,openai_api_key=OPENAI_API_KEY)
    seV2 = SearchEngineWithLLM(base_llm=llm, search_engine=se)
    return seV2


if __name__ == "__main__":
    se = build_search_engine()
    while True:
        term = input("Prompt: ")
        print(se.search(term))
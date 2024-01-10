from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document
from langchain.vectorstores import Weaviate

from utils.weaviate_client import client

CATEGORIES = [
    "Action",
    "Documentary",
    "Family",
    "Drama",
    "Horror",
    "Fantasy",
    "Adventure",
    "History",
    "Romance",
    "Music",
    "Western",
    "Animation",
    "War",
    "Comedy",
    "Mystery",
    "TV Movie",
    "Thriller",
    "Science Fiction",
    "Crime",
]

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="genres",
        description="The genres of the movie. Must be one of the following: {', '.join(CATEGORIES)}}",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="release_year",
        description="The year the movie was released",
        type="float",
    ),
    AttributeInfo(
        name="imdb_vote_average",
        description="A 1-10 rating for the movie",
        type="float",
    ),
    AttributeInfo(
        name="imdb_vote_count",
        description="The number of reviews the movie has on IMDB",
        type="float",
    ),
]


def get_best_docs(input: str, providers: list[int]) -> list[Document]:
    document_content_description = "Brief summary of a movie"
    llm = OpenAI(temperature=0)

    vectorstore = Weaviate(
        client,
        "Movie",
        "text",
        attributes=[
            "title",
            "release_year",
            "runtime",
            "genres",
            "imdb_vote_count",
            "imdb_vote_average",
            "trailer_url",
            "watch",
        ],
    )
    
    where_filter = {
        "path": ["providers"],
        "operator": "ContainsAny",
        "valueNumber": [int(p) for p in providers],
    }
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        METADATA_FIELD_INFO,
        verbose=True,
        search_kwargs={"k": 3, "where_filter": where_filter},
    )
    
    return retriever.get_relevant_documents(input)

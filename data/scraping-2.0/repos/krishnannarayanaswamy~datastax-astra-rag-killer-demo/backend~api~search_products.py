import os
from backend.api.datamodel import SearchResult
from backend.api.config import (
    STYLE_MAPPING,
    SEASON_COLUMNNAME,
    GENDER_COLUMNNAME,
)
from typing import List, Dict
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB

load_dotenv()

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
openai_api_key=os.environ["OPENAI_API_KEY"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="ecommerce_inventory",
    api_endpoint=api_endpoint,
    token=token,
)

def search(
    query: str,
    filters: str,
    gender: str,
    personal_preferences: str,
    favourites: List[str],
    style: str = None,
    main_term: str = None,
) -> List[SearchResult]:

    filterresults = {}
    style_modifier = STYLE_MAPPING.get(style)

    if filters:
        filterresults[GENDER_COLUMNNAME] = filters

    if gender:
        filterresults[GENDER_COLUMNNAME] = gender

    if query:
        main_term = query

        if not style_modifier:
            main_term = query
        elif style_modifier:
            main_term = style_modifier.replace("<QUERY>", query)
        
        if personal_preferences:
            main_term = f"{main_term} {personal_preferences}"

    print(f"Astra Query term: {main_term}")
    print(f"Astra filter: {filterresults}")
    results=vstore.similarity_search(main_term, k=10, filter=filterresults)

    return [
        SearchResult(
            id=r.metadata.get("title"),
            name=r.metadata.get("basename"),
            price=float(r.metadata.get("price")),
            image_url=r.metadata.get("s3_http"),
        )
        for r in results
    ]

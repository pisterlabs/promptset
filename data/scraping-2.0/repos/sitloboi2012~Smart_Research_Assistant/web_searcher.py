from functools import lru_cache
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults

import requests
import streamlit as st

"""
db = ZillizVectorDatabase()
langchain_db = Zilliz(
    embedding_function = EMBEDDING_FUNC,
    collection_name = "Production",
    connection_args = {
        "uri": db.cloud_uri,
        "token": db.cloud_api_key,
        "secure": True,
    },
    consistency_level = "Session",
    primary_field="pk",
    text_field="document",
    vector_field="vector"
)

ziliz_retriever = ZillizRetriever(
    embedding_function = EMBEDDING_FUNC,
    collection_name = "Production",
    connection_args = {
        "uri": db.cloud_uri,
        "token": db.cloud_api_key,
        "secure": True,
    },
    consistency_level = "Session",
    primary_field="pk",
    text_field="document",
    vector_field="vector",
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    #search_params={"search_type": "mmr", "param": {"lambda": 0.5, "k": 10}}
)

"""
ddg_search = DuckDuckGoSearchAPIWrapper(max_results = 100)
ddg_tool = DuckDuckGoSearchResults(api_wrapper=ddg_search, max_results = 100)

query = "XR in Business Applications"
description = """
Embark on a captivating exploration of Extended Reality (XR) in marketing through this 10-week project. In the first three weeks, you will concentrate on a guided deep dive into the literature of XR, which encompasses Augmented Reality (AR), Virtual Reality (VR), and Mixed Reality (MR). 
This phase will focus on understanding the concepts and potential applications of XR in marketing.Weeks four and five will involve analyzing the literature to identify key trends and patterns in XR marketing. 
This analysis will enhance your understanding of how XR has been evolving and what aspects of marketing it has been impacting.
In weeks six to eight, you will shift your focus to examining real-world case studies. 
Select three impactful XR marketing campaigns and investigate them in depth. Evaluate the technologies used, the response from consumers, and the overall effectiveness of the campaigns. 
During the ninth week, you will synthesize the knowledge gained from the literature and case studies to develop insights into the current state of XR in marketing. 
In the final week, you will compile your insights and analyses into a comprehensive report. 
This report will document your journey through the XR literature, key trends, and real-world applications.
"""

@lru_cache 
def web_search(query: str, num_results: int):
    results = ddg_search.results(query, num_results)
    return [{"link": r["link"], "title": r["title"]} for r in results]

def search_paper(keyword: str, field_of_study: str):
    data = {
        "query": keyword,
        "fieldsOfStudy": field_of_study,
        "fields": "title,year,authors,abstract,citationCount,references,citations,s2FieldsOfStudy,url,publicationDate,journal,referenceCount,citationStyles,fieldsOfStudy",
        "limit": 20
    }
    BASE_URL = f"https://api.semanticscholar.org/graph/v1/paper/search?query={data['query']}&fieldsOfStudy={data['fieldsOfStudy']}&fields={data['fields']}&limit={data['limit']}"
    payload = {}
    headers = {
        'x-api-key': st.secrets["SEMANTIC_SCHOLAR_API"]
    }

    response = requests.request("GET", BASE_URL, headers=headers, data=payload)
    
    return response.json()


"""
question_generator = QuestionGenerator()
output = question_generator.generate_question(query, description)
final_response = question_generator.filter_result(query, description, output)
keyword_list = question_generator.parsing_keyword(final_response)
print("--------- \n")
print(keyword_list)

def search_paper(keyword: str, field_of_study: str):
    data = {
        "query": keyword,
        "fieldsOfStudy": field_of_study,
        "fields": "title,year,authors,abstract,citationCount,publicationTypes,references,citations,fieldsOfStudy,s2FieldsOfStudy"
    }
    BASE_URL = f"https://api.semanticscholar.org/graph/v1/paper/search?query={data['query']}&fieldsOfStudy={data['fieldsOfStudy']}&fields={data['fields']}"
    payload = {}
    headers = {}

    response = requests.request("GET", BASE_URL, headers=headers, data=payload)
    
    return response.json()

for i in keyword_list:
    input_query = i.split(".")[-1].strip()
    search_result = search_paper(input_query, "Business,Economics,Education,Linguistics,Engineering,Political Science,Sociology,Computer Science,Psychology")
    if search_result["total"] > 0:
        with open("output.txt", "a") as f:
            f.write(search_result["data"][0]["title"])
            f.write("\n")
            f.write(search_result["data"][0]["abstract"])
            f.write("\n")
            f.write("--------- \n")
"""
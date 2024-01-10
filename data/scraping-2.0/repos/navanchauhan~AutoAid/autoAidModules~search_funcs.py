import os

from serpapi import GoogleSearch
from .sample_res import res
from boilerpy3 import extractors
from fake_useragent import UserAgent
from langchain.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate

import requests

extractor = extractors.ArticleExtractor()

preferred_forums = {
    "BMW": ["bimmerforums.com"],
    "Subaru": ["nasioc.com"]
}

llm = Bedrock(model_id="anthropic.claude-instant-v1")
ua = UserAgent()

"""
Website data: 
[

{
    "title":"",
    "link": "",
    "date": "", # prioritise older posts for older cars?,
    "full-text": "",
},

]
"""

def find_preferred_forums(make):
    if make not in preferred_forums:
        template = "Human:  If BMW: bimmerforums.com, Subaru: nasioc.com, Mazda: forum.miata.net What is the best forum for {make}? No more explanation\n\nAssistant: Then {make}:"
        prompt = PromptTemplate(input_variables=["make"], template=template)
        pred = llm.predict(prompt.format(make=make), max_tokens_to_sample=30, temperature=1,top_k=250, top_p=0.999)
        make_url = pred.strip().split()[0]
        print(f"Found {make_url} for {make}")
        preferred_forums[make] = [make_url]
    return preferred_forums[make]

def get_preferred_forums(make):
    if make not in preferred_forums:
        return find_preferred_forums(make)
    return preferred_forums[make]

def parse_page(url):
    content = extractor.get_content_from_url(url)
    return content

def get_tasks_from_pages(pages: list = [], query: str = "", details: str = ""):
    template =  "Human: You are an beginner mechanic. You are trying to solve the problem of {query} and have a {details}.\n Generate simple tasks from the following pages:\n {pages}\n\nAssistant: I would try all of the following, one by one:\n\n- Have you tried turning your car on and off?\n- "
    prompt_template = PromptTemplate(input_variables=["query", "details", "pages"], template=template)

    
    pred = llm.predict(
        prompt_template.format(
            query=query, details=details, pages=pages
                ), max_tokens_to_sample=501, temperature=1,top_k=250, top_p=0.999
                    )
    pred = "- " + pred
    print(pred)
    return pred


def search_on_forum(forum, query, max_results: int = 5):
    params = {
        "q": query + f" {forum}",
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": os.environ.get("SERP_API_KEY", "demo")
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    #results = res # Debugging Data
    if results["search_metadata"]['status'] == "Success":
        data = []
        for idx, result in enumerate(results["organic_results"]):
            if idx >= max_results:
                break
            new_dict = {
                "title": result["title"],
                "link": result["link"],
                "full-text": ""
            }
            try:
                resp = requests.get(result["link"], headers={"User-Agent": ua.random})
                new_dict["full-text"] = extractor.get_content(resp.text)
            except Exception as e:
                print(f"Error parsing page {result['link']}: {e}")
            data.append(new_dict)
        return data
    else:
        return []

import os
import re
import base64
import requests
import json
import logging
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.agents import Tool
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
nltk_download('stopwords')
nltk_download('punkt')
from nltk.corpus import stopwords

# Check for required ENV variables
if(os.environ.get("CONFLUENCE_CLIENT_ID") == None):
    msg = "CONFLUENCE_CLIENT_ID not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

if(os.environ.get("CONFLUENCE_API_KEY") == None):
    msg = "CONFLUENCE_API_KEY not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

if(os.environ.get("CONFLUENCE_SITE_NAME") == None):
    msg = "CONFLUENCE_SITE_NAME not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

# Splits strings into chunks of X length
def split_str(seq, chunk, skip_tail=False):
    lst = []
    if chunk <= len(seq):
        lst.extend([seq[:chunk]])
        lst.extend(split_str(seq[chunk:], chunk, skip_tail))
    elif not skip_tail and seq:
        lst.extend([seq])
    return lst

# The function that the tool executes - this one searches confluence
def search_confluence(query) -> str:
    # Replace these variables with your own values
    atlassian_email = os.environ.get("CONFLUENCE_CLIENT_ID")
    api_token = os.environ.get("CONFLUENCE_API_KEY")
    your_site_name = os.environ.get("CONFLUENCE_SITE_NAME")

    # Encode the email and API token in base64 format
    credentials = "{atlassian_email}:{api_token}".format(atlassian_email=atlassian_email, api_token=api_token)
    encoded_credentials = base64.b64encode(bytes(credentials, 'UTF-8')).decode()

    # Set the base URL and the search query parameters
    base_url = f"https://{your_site_name}.atlassian.net/wiki/rest/api"
    search_endpoint = "/search"
    params = {
        "limit": 2,
        "cql": "title ~ '{query}' or text ~ '{query}'".format(query=query),
        "space": "SNB",
    }

    # Set the headers, including the encoded credentials for authorization
    headers = {
        "Accept": "application/json",
        "Authorization": "Basic {encoded_credentials}".format(encoded_credentials=encoded_credentials)
    }

    # Make the GET request to the Confluence API
    response = requests.get(base_url + search_endpoint, headers=headers, params=params)

    # Check if the request was successful (HTTP status code 200) and build YAML response obj
    if response.status_code == 200:
        results = response.json()
        return_str = "Search Results for '{query}':\n".format(query=query)
        res_counter = 0
        for result in results["results"]:
            if "content" in result:
                if result["content"]["type"] == "page":
                    res_counter += 1
                    title = result["content"]["title"]
                    page_url = result["content"]["_links"]["self"]
                    params = {"expand": "body.storage"}
                    page_request = requests.get(page_url, headers=headers, params=params)
                    body_html = page_request.json()["body"]["storage"]["value"]
                    re_obj = re.compile(r'<ac.*?/>')
                    body_html = re_obj.sub("", body_html)
                    body_text = BeautifulSoup(body_html, "html.parser")
                    body_text = "\n".join(body_text.text.split("\n"))
                    tokens = word_tokenize(body_text.lower())
                    english_stopwords = stopwords.words('english')
                    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]
                    body_text = " ".join(tokens_wo_stopwords)
                    if len(body_text) > 2000:
                        body_text = split_str(body_text, 2000, True)[0]
                    summary_llm = OpenAI(max_tokens=500)
                    body_text = summary_llm("Re-write the following text into a single large paragraph, retaining as much information as possible: {body_text}".format(body_text=body_text))
                    result_str = " - title: {title}\n   body_text: '{body_text}'\n".format(title=title, body_text=body_text)
                    return_str = "{}{}".format(return_str, result_str)
        if res_counter > 0:
            return return_str
        else:
            logging.info("No confluence results for: {query}".format(query=query))
            return "No search results found - use another tool to find an answer for the user"
    else:
        logging.warn("Confluence search request failed with status code {response_code}".format(response_code=response.status_code))
        return "Request failed with status code {response_code}".format(response_code=response.status_code)

    
# Define the tool as LangChain tool objects for agent use
SearchConfluenceTool = Tool(
    name = "Search the knowledgebase",
    func = search_confluence,
    description="Use this tool first. Useful for searching the users private confluence knowledgebase for information, articles or documents. Input should be key words to search for."
)
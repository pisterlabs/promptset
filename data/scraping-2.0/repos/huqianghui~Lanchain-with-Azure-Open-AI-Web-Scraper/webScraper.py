from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI
from datetime import datetime
import os
from autogen import config_list_from_json
import autogen

import requests
from bs4 import BeautifulSoup
import json

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import openai
from dotenv import load_dotenv

openai.api_version = os.environ.get("OPENAI_API_VERSION")
openai.log = os.getenv("OPENAI_API_LOGLEVEL")
openai.api_type == "azure"


def web_qa(url_list, query, out_name):
    openai = ChatOpenAI(
        model_name="gpt-4",
        max_tokens=2048
    )
    loader_list = []
    for i in url_list:
        print('loading url: %s' % i)
        loader_list.append(WebBaseLoader(i))

    index = VectorstoreIndexCreator().from_loaders(loader_list)
    ans = index.query(question=query,
                      llm=openai)
    print("")
    print(ans)

    outfile_name = out_name + datetime.now().strftime("%m-%d-%y-%H%M%S") + ".out"
    with open(outfile_name, 'w') as f:
        f.write(ans)

# url_list = [
#     "https://openaimaster.com/how-to-use-ideogram-ai/",
#     "https://dataconomy.com/2023/08/28/what-is-ideogram-ai-and-how-to-use-it/",
#     "https://ideogram.ai/launch",
#     "https://venturebeat.com/ai/watch-out-midjourney-ideogram-launches-ai-image-generator-with-impressive-typography/"
# ]

prompt = '''
    Given the context, please provide the following:
    1. summary of what it is
    2. summary of what it does
    3. summary of how to use it
    4. Please provide 5 interesting prompts that could be used with this AI.
'''

# web_qa(url_list, prompt, "summary")

# def scaper(url_list):
#     loader_list=[]
#     for i in url_list:
#         print('loading url: %s' % i)
#         doc = WebBaseLoader(i).load()
#         print(doc)

# url_list = ["https://www.amazon.com.au/dp/B0851B8QJ7/"]
# scaper(url_list)

def scrape(url: str,fileName:str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        "https://chrome.browserless.io/content?token=043abdc1-b765-4298-b7b8-b2c39f6ce27e", headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        with open(fileName, 'w') as f:
            print(" length of the fileName:\t" + fileName + "  lenght:\t" + str(len(text)))
            f.write(text)

        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# result = scrape("https://www.amazon.com.au/dp/B0851B8QJ7/")


def scrapeReviewList():
    base_url = "https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="
    for i in range(1, 11):
        url = base_url + str(i)
        print(url)
        scrape(url, "review"+str(i)+".txt")

scrapeReviewList()      
#result = scrape("https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
# https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2
# https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_getr_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber=3
# https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_getr_d_paging_btm_next_4?ie=UTF8&reviewerType=all_reviews&pageNumber=4
# https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_getr_d_paging_btm_next_9?ie=UTF8&reviewerType=all_reviews&pageNumber=9
# https://www.amazon.com.au/FLEXISPOT-Electric-Standing-Adjustable-Workstation/product-reviews/B0851B8QJ7/ref=cm_cr_getr_d_paging_btm_next_10?ie=UTF8&reviewerType=all_reviews&pageNumber=10
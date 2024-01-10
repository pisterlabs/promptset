# ### Has dictionary that you get of categories and headlines from each. Aim is to use NLP to summarize headline for keywords (4 word summary), search keywords using news api, scrape news api using llm, summarize different stories
import openai
import re
from categorizer import *
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob
from langchain.document_loaders import JSONLoader


# Initialize the OpenAI API client
openai.api_key = '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'

article_summaries = {} # by category, by headline (dictionary): summary, two major perspective summaries, external links

# Parse through dictionary: 
for category in mapping:
    for article in category: 
        search = summarize_headline_5_words(article)
        test1 = "US Politician Detained Hong Kong"
        
        # APPROACH 1 : USING NEWSAPI, Collecting top articles and putting them in a dictcionary, then for each article, parsing over the json using Langchain and summarizing headline
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": search,
            "from": "2023-10-16",
            "sortBy": "popularity",
            "apiKey": "785379e735a84282af1c6b35cf335a59"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        headline = f"S"
        obj_summary = 
        p1_summary = 
        p2_summary = 

        print(data)
        # storing article and hyperlinks as json in data 
        # DO WE REALLY NEED THIS  -- THIS IS APPROACH 1
        mapping[category][article][json] = data 

        # APPROACH #2 -- given article summary, searching up summarized keywords, and coming up with headlines 








def summarize_headline_5_words(article):
    """Summarizes the article headline into 4 words.

    Args:
        article: A string variable containing the article headline.

    Returns:
        A 5-word summary of the article headline.
    """
    message = {
        "role": "system",
        "content": "You are a helpful assistant. Summarize headlines into 4 words."
    }
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            message,
            {"role": "user", "content": f"Please summarize this headline: {article}"}
        ]
    )
    
    # Extracting the summarized text from the response.
    summarized_text = response['choices'][0]['message']['content'].strip()
    return summarized_text

# Get the article headline
article_headline = "Off-duty pilot accused of trying to crash Alaska Airlines jet cites breakdown"

# Print the summary
print(summarize_headline_5_words(article_headline))


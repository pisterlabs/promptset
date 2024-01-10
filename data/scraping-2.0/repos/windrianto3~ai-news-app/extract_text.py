from datetime import date
from dotenv import load_dotenv
import os
import json

import time
import requests as req
from bs4 import BeautifulSoup

load_dotenv()
API_KEY=os.environ.get('NYT_API_KEY')

def _getDate():
    today = date.today()	
    today_date = today.strftime("%Y%m%d")
    return today_date

def scrape_data(topic, num_pages):
    articles = {}
    today = _getDate()
    for i in range(num_pages):
        url='https://api.nytimes.com/svc/search/v2/articlesearch.json?q='+topic+'&begin_date='+today+'&end_date='+today+'&api-key='+API_KEY+'&page='+str(i)
        response = req.get(url).json()
        # Extract the necessary fields from the response.
        docs = response['response']['docs']
        for doc in docs:
            filteredDoc = {}
            filteredDoc['title'] = doc['headline']['main']
            filteredDoc['abstract'] = doc['abstract']
            filteredDoc['paragraph']=doc['lead_paragraph']
            filteredDoc['url'] = doc['web_url']
            filteredDoc['body'] = ''
            filteredDoc['summary'] = ''
            articles[doc['web_url']] = filteredDoc
        # Done to avoid hitting the API request limit.
        time.sleep(6)
        scrape_text(articles)
    return articles

def scrape_text(article_data):
    # Loop through the article URLs
    for key in article_data.keys():
        try:
            # Send an HTTP GET request to the URL
            response = req.get(key, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64"})

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML content of the page using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find the <article> tag, assuming it contains the main content
                article = soup.find('article')

                # Initialize a variable to store the cleaned text
                cleaned_text = ""

                # Loop through the paragraphs within the <article> tag
                for paragraph in article.find_all('p'):
                    cleaned_text += paragraph.get_text() + "\n"

                # Store the URL and cleaned text in the dictionary
                article_data[key]['body'] = cleaned_text
            else:
                print(f"Failed to retrieve content from {key}. Status code: {response.status_code}")

        except Exception as e:
            print(f"Error fetching content from {key}: {str(e)}")

def summarize_articles(article_data):
    for key in article_data.keys():
        # Call LLM summarizer
        text_to_summarize = article_data[key]['body'][:15700]
        summarized = summarize(text_to_summarize)
        article_data[key]['summary'] = summarized
        time.sleep(21)
        
from langchain import OpenAI
from langchain import PromptTemplate
# from langchain.utilities import WikipediaAPIWrapper

openai_api_key = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

def summarize(body):
    # Call the AI here to summarize the passed in body text
    template = """
    Please write a brief single paragraph summary (at most 100 words) of the following New York Times article:

    {article}
    """
    prompt = PromptTemplate(
    input_variables=["article"],
    template=template
    )
    
    summary_prompt = prompt.format(article = body)
    summarized = llm(summary_prompt)
    
    return summarized.strip()
        
def _test():
    # Create a dictionary to store URL-Doc object pairs
    # article_data[URL] gives a filtered Doc object. article_data[URL]['body'] gives body paragraph
    article_data = scrape_data("Technology", 1)

    scrape_text(article_data)
    summarize_articles(article_data)
    
    # Now, you have a dictionary containing URL-object pairs
    # You can access or process this data as needed
    #print(article_data)
    with open('data.json', 'w') as fp:
        json.dump(article_data, fp, indent=4)
    
def generateDailyReport(topic):
    article_data = scrape_data(topic, 1)
    scrape_text(article_data)
    summarize_articles(article_data)
    
    # Python dictionary consisting of ArticleURL:object mappings
    return article_data
    
# _test()
    
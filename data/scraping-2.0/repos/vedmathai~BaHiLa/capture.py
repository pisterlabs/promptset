from datetime import datetime
import sys
# Add the path of the directory containing the theguardian module to sys.path
# Replace 'path_to_theguardian_module' with the actual path to the module's directory
sys.path.append('theguardian-api-python')
from theguardian import theguardian_content
import requests
import openai
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

PAGESIZE = 50

# Filter function to remove unwanted content
unwanted_keywords = ["football", "society","fashion","music","lifeandstyle","environment","media","tv-and-radio","film"]
def filter_function(content):
    title = content.get('webTitle', '').lower()
    section = content.get('sectionName', '').lower()
    for keyword in unwanted_keywords:
        if keyword in title or keyword in section:
            return False
    return True

def get_content(fromdate, todate, keyword, api_key):
    """
    Fetches and filters articles from The Guardian based on given criteria.
    
    Parameters:
    - fromdate (str): The start date in the format "YYYY-MM-DD" to fetch articles from. 
    - todate (str): The end date in the format "YYYY-MM-DD" to fetch articles until.
    - keyword (str): A keyword or phrase to search within the articles.
                     If left empty, it will fetch articles without a specific keyword filter.
    - api_key (str): Your API key provided by The Guardian.

    Returns:
    - List[Dict]: A filtered list of articles from The Guardian. Each article 
                  is represented as a dictionary with details like title, URL, etc.

    Description:
    The function fetches articles from The Guardian's "world" section within 
    the specified date range and containing the given keyword. The fetched articles 
    are then filtered to exclude those with certain unwanted keywords or from unwanted sections.
    """
    
    # Set up query parameters
    params = {
        "section" : "world",
        "from-date": fromdate,
        "to-date": todate,
        "q": keyword,
        "api-key": api_key,
        "page-size": PAGESIZE
    }


    
    # Create the Content instance with the specified parameters
    content = theguardian_content.Content(api=api_key, **params)

    # Get content response
    json_content = content.get_content_response()

    # Extract results from the response
    results = content.get_results(json_content)

    # Apply the filter to the results
    filtered_results = list(filter(filter_function, results))
    
    return filtered_results


def fetch_article_content(api_key, endpoint):
    # Construct the full URL with the API key
    url = f"{endpoint}?api-key={api_key}&show-fields=body"
    
    # Make the GET request to the API
    response = requests.get(url)
    
    # If the request was successful, extract and return the article content
    if response.status_code == 200:
        data = response.json()
        article_content = data["response"]["content"]["fields"]["body"]
        article_content = BeautifulSoup(article_content, 'html.parser').get_text()
        return article_content
    else:
        return f"Error {response.status_code}: Unable to fetch the article."
    
def event_type_classification(headline):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": f"Here is the title of a news piece: {headline}\nQuestion: Is this a tragic event? Only answer yes or no."
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    return response["choices"][0]['message']['content']


def get_title_and_content(start_time, end_time, keyword, api_key):
    contents = get_content(start_time, end_time, keyword, api_key)
    news_dic = dict()
    for item in contents:
        date = item.get('webPublicationDate','').split('T')[0]
        endpoint = item.get('apiUrl','')
        item = {"title": item.get('webTitle', ''), "endpoint": item.get('apiUrl',''),
        "main_content": fetch_article_content(api_key, endpoint), "url": item.get('webUrl','')}
        if date not in news_dic:
            news_dic[date] = [item]
        else:
            news_dic[date].append(item)
    return news_dic



def filter_news(start_time, end_time, keyword, api_key):
    news_dic = get_title_and_content(start_time, end_time, keyword, api_key)
    # print(news_dic)
    filtered_dic = dict()
    for date, data in news_dic.items():
        filtered_dic[date] = list()
        for news in data:
            if 'yes' in event_type_classification(news['title']).lower():
                filtered_dic[date].append(news)
    
    return filtered_dic


def get_casualty_dict(main_content):
    # Schema
    schema = {
        "properties": {
            "event_subject": {"type": "string"},
            "event_number_of_victims": {"type": "integer"},
            "event_location": {"type": "string"},
        },
        # "required": ["number_of_victims"],
    }

    # Input (trimmed)
    inp = main_content[:4000]

    # Run chain
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = create_extraction_chain(schema, llm)
    result = chain.run(inp)

    final_result = None
    for event in result:
        # make sure the event_number_of_victims is not None
        if event['event_number_of_victims'] is None:
            continue
        if event['event_number_of_victims'] == max([e['event_number_of_victims'] for e in result if e['event_number_of_victims'] is not None]):
            final_result  = event
    
    return final_result
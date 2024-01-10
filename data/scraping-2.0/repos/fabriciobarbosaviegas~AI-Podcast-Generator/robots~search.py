import openai
from time import sleep
import requests
from bs4 import BeautifulSoup
import simple_colors  
from api_config import setBraveAPI



def search(topic):
    terms = getSearchTerms(topic)
    results = [{"term":term,"searchResults":getSearchResults(term)} for term in terms]
    
    print(simple_colors.blue(f"\n\n================== Results ===============\n\n"))

    for result in results:
        print(simple_colors.blue(f"{result}\n"))

    return results



def getSearchTerms(topic):
    searchTerms = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a robot specialized in searching on Bing for relevant articles on any topic provided to you, from a user defined topic you are able to list related search terms capable of leading to high credibility and converting articles on the given subject. You always return only an unnumbered list of terms separated by '* ' without personal comments."},
            {"role": "user", "content": topic},
        ]
    )
    response = searchTerms.choices[0].message["content"]

    searchTerms = parseSearchTerms(response)
    searchTerms.append(topic)

    print(simple_colors.blue(f"\nsearch terms: {searchTerms}\n"))

    return searchTerms



def parseSearchTerms(message):
    text = message.replace("\n", "")
    text = text.split("*")[1:]
    return text
    


def getSearchResults(term):
    print(simple_colors.blue(f"Searching for '{term}'..."))
    sleep(2)
    links = []
    headers={"Accept":"application/json", "Accept-Encoding":"gzip", "X-Subscription-Token":setBraveAPI()}

    target_url = f"https://api.search.brave.com/res/v1/web/search?q={term}&country=US&result_filter=web"

    results = requests.get(target_url, headers=headers).json()['web']['results']

    for url in results[:5]:
        links.append(url['url'])

    return links
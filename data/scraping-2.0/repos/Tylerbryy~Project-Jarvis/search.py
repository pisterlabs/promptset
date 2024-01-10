import webbrowser
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import openai
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPEN_AI_APIKEY")

def search_and_scrape(query):
    q=[]

    q.append({"role": "assistant", "content": f"try to summarize what i am saying into a search query. your response should be only the search query nothing else before or after it"})
    q.append({f"role": "user", "content": f"{query}"})

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=q
    )

    query_summary = response.choices[0].message.content
    print(query_summary)
    
    
    messages = []
    
    # Search the web
    for url in search(query_summary, num_results=1):
        # Get the webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        
        text = ''
        for p_tag in soup.find_all('p'):
            text += p_tag.get_text()
            
        print(text)
        
        messages.append({"role": "assistant", "content": f"You are professional summarizer"})
        messages.append({f"role": "user", "content": f"Summarize the following text in 80 words : {text}"})
        
        webbrowser.open(url)

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    
        summary = response.choices[0].message.content
        
        return summary
        



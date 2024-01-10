import requests
from bs4 import BeautifulSoup
import webbrowser
import openai_secret_manager
import openai
import os
import subprocess
from key import chatgpt_key
from stem import Signal
from stem.control import Controller



script_path = 'login.py'
# Execute the script with a buffer size of 1024
subprocess.run(['python3', script_path], bufsize=1024)

def tor_search():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate()
        controller.signal(Signal.NEWNYM)
        session = requests.session()
        session.proxies = {'http': 'socks5h://localhost:9050', 'https': 'socks5h://localhost:9050'}
        search_query = input("Enter your search query: ")
        url = f"https://www.google.com/search?q={search_query}"
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        result = soup.prettify()
        action = input("Do you want to print the result, save it to an HTML file or open it in a browser? (Enter 'print', 'save' or 'open'): ")
        if action == 'print':
            print(result)
        elif action == 'save':
            with open('tor_results.html', 'w') as f:
                f.write(result)
        elif action == 'open':
            with open('tor_results.html', 'w') as f:
                f.write(result)
            webbrowser.open('result.html')

# Define the available search engines
search_engines = {

    'Ask.com': 'https://www.ask.com/web?q=',
    'Brave': 'https://search.brave.com/search?q=',
    'Dogpile': 'https://www.dogpile.com/search/web?q=',
    'DuckDuckGo': 'https://duckduckgo.com/html/?q=',
    'Ecosia': 'https://www.ecosia.org/search?q=',
    'Exalead': 'https://www.exalead.com/search/web/results/?q=',
    'Gigablast': 'https://www.gigablast.com/search?q=',
    'Google': 'https://www.google.com/search?q=',
    'Microsoft Bing': 'https://www.bing.com/search?q=',
    'Qwant': 'https://www.qwant.com/?q=',
    'ChatGPT': '',
    'Tor': ''
}

# Get the user's choice of search engine
engine_choice = input(f"Which search engine would you like to use? ({', '.join(search_engines.keys())}): ")

# Get the user's search term
search_term = input("What would you like to search for? ")

if engine_choice == "ChatGPT":
    # Send the search term to ChatGPT
    assert "openai" in openai_secret_manager.get_services()
    secrets = openai_secret_manager.get_secret("openai")
    
    openai.api_key = secrets[chatgpt_key]
    
    response = openai.Completion.create(
        engine="davinci",
        prompt=search_term,
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    

    print(response['choices'][0]['text'].strip())
elif engine_choice == "Tor":
    tor_search()
else:
    # Perform the search using a search engine
    search_url = search_engines[engine_choice] + search_term
    response = requests.get(search_url)

    # Get the user's choice of how to handle the results
    result_choice = input("How would you like to handle the results? (1: Open in web browser, 2: Save as HTML file): ")

    if result_choice == "1":
        # Open the search results in the default web browser
        webbrowser.open(search_url)
    elif result_choice == "2":
        # Save the search results as an HTML file in the INPUT folder
        if not os.path.exists('INPUT'):
            os.makedirs('INPUT')
        with open(os.path.join('INPUT', 'search_results.html'), 'a') as f:
            f.write(response.text)            
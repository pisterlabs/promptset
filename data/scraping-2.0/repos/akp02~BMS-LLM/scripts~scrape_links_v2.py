from langchain.document_loaders import UnstructuredHTMLLoader, SeleniumURLLoader
from bs4 import BeautifulSoup
import requests
import json
#get all links from bmsce.ac.in domain
def get_all_links(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return links
    except requests.RequestException as e:
        print(f"Error: {e}")
        return []

#return all links from bmsce.ac.in domain
def crawl_website(starting_url, depth=3):
    visited_urls = set()
    bmsce_urls = []

    def crawl(url, current_depth):
        if current_depth > depth:
            return
        if url in visited_urls:
            return

        visited_urls.add(url)
        #check if url contains bmsce.ac.in
        if "bmsce.ac.in" in url:
            bmsce_urls.append(url)
        # print(f"Crawling: {url}")
        

        links = get_all_links(url)
        for link in links:
            crawl(link, current_depth + 1)

    crawl(starting_url, 1)
    return bmsce_urls

#dump bmsce_urls to a json file
def dump_to_json(bmsce_urls):
    with open("bmsce_urls.json", "w") as f:
        json.dump(bmsce_urls, f, indent=2)

#scrape all links from a list using seleniumUrlLoader
def scrape_links(links_list):
    data = []

    for link in links_list:
        loader = SeleniumURLLoader(urls = [link])
        document = loader.load()
        print(document)
        metadata = document[0].metadata

        entry = {
            'metadata': metadata,
        }
        try:
            entry['text'] = document[0].page_content
        except Exception as e: 
            print(f"Error: {e} while scraping text of {link}")
            entry['text'] = None
        data.append(entry)
    return data



if __name__ == "__main__":
    #crawl website
    
    bmsce_urls = crawl_website("https://bmsce.ac.in")
    dump_to_json(bmsce_urls)
    #load links from bmsce_urls.json
    with open("bmsce_urls.json", "r") as f:
        bmsce_links = json.load(f)
    data = scrape_links(bmsce_links)
    
    #save to json
    with open("bmsce_data_langchain.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Done!")
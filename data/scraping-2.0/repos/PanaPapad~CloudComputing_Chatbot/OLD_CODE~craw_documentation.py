# from IPython.display import clear_output"""  """
from urllib.parse import urljoin
import pickle 
urls = "https://ucy-linc-lab.github.io/fogify/"

from langchain.document_transformers import Html2TextTransformer
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader

def crawlulr(start_url):
  url_contents=[]
  # Send a GET request to the URL
  response = requests.get(start_url)

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the HTML content
      soup = BeautifulSoup(response.text, 'html.parser')

      # Find all 'a' tags (links) in the page
      links = soup.find_all('a')

      # Extract and print all URLs
      for link in links:
          # Get the href attribute of each 'a' tag
          href = link.get('href')

          # Join the URL if it's relative
          full_url = urljoin(start_url, href)

          print(f"Crawling url: {full_url}")
          loader = AsyncHtmlLoader(full_url)
          docs = loader.load()
          html2text = Html2TextTransformer()
          docs_transformed = html2text.transform_documents(docs)

          url_contents.append(docs_transformed)
  else:
      print("Failed to retrieve the page")
  write_list(url_contents)

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('/data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
def write_list(a_list):
    # store list in binary file so 'wb' mode
    with open('/data/urlcontent.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')
crawlulr(urls)
print(len(read_list()))
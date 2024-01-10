import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

# # Regex Pattern to match a URL
# HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# # Define Root domain to crawl
# domain = "openai.com"
# full_url = "https://openai.com/"

# # Create a class to parse the HTML and get the hyperlinks
# class HyperLinkParser(HTMLParser):
#     def __init__(self):
#         super().__init__()
#         # Create a list to store the hyperlinks
#         self.hyperlinks = []
    
#     # Override the HTMLParser's handle_starttag method to get the hyperlinks
#     def handle_starttag(self, tag, attrs):
#         attrs = dict(attrs)

#         # If tag is an anchor tag and it has an href attribute, add the href attribute to the list of the hyperlinks
#         if tag == "a" and "href" in attrs:
#             self.hyperlinks.append(attrs["href"])

# # Function to get hyperlinks from a URL
# def get_hyperlinks(url):

#     # Try to oprn the URL and read the HTML:
#     try:
#         with urllib.request.urlopen(url) as response:

#             # If the response is not HTML return an empty list
#             if not response.info().get('Content-Type').startswith("text/html"):
#                 return []

#             # Decode the HTML
#             html = response.read().decode('utf-8')
#     except Exception as e:
#         print(e)
#         return []

#     # Create the HTMLParser and then Parse the HTML to get hyperlinks
#     parser = HyperLinkParser()
#     parser.feed(html)

#     return parser.hyperlinks

# # Function to get the hyperlinks from a URL that are within the same domain
# def get_domain_hyperlinks(local_domain, url):
#     clean_links = []

#     for link in set(get_hyperlinks(url)):
#         clean_link = None

#         # If the link is a URL, check if it is within the same domain
#         if re.search(HTTP_URL_PATTERN, link):
#             # Parse the URL and check if the domain is the same
#             url_obj = urlparse(link)
#             if url_obj.netloc == local_domain:
#                 clean_link = link
        
#         # If the link is not a URL, check if it is a relative link
#         else:
#             if link.startswith("/"):
#                 link = link[1:]
#             elif link.startswith("#") or link.startswith("mailto: "):
#                 continue
#             clean_link = "https://"+local_domain+"/"+link
            
#         if clean_link is not None:
#             print("clean_link: "+clean_link)
#             if clean_link.endswith("/"):
#                 clean_link = clean_link[:-1]
#             clean_links.append(clean_link)
    
#     # Return the list of hyperlinks that are within the same domain
#     return list(set(clean_links))

# def crawl(url):
#     # Parse the URL and get the domain
#     local_domain = urlparse(url).netloc
    
#     # Create a queue to store the URLs to crawl
#     queue = deque([url])
    
#     # Create a set to store the URLs that have already been seen (no duplicate)
#     seen = set([url])
    
#     # Create a directory to store the text files
#     if not os.path.exists("text/"):
#         os.mkdir("text/")
    
#     if not os.path.exists("text/"+local_domain+"/"):
#         os.mkdir("text/"+local_domain+"/")
        
#     # Create a directory to store the csv files
#     if not os.path.exists("processed"):
#         os.mkdir("processed")
        
#     # While queue is not empty
#     while queue:
        
#         # Get the next URL from the queue
#         url = queue.pop()
#         print(url) # for debugging and to see the programs
        
#         with open('text/'+local_domain+'/'+url[8:].replace('/',"_")+".txt","w", encoding="utf-8") as f:
            
#             # Get the text from the URL using BeautifulSoup
#             soup = BeautifulSoup(requests.get(url).text, "html.parser")
            
#             # Get the text but remove the tags
#             text = soup.get_text()
            
#             # If the crawler gets to a page that requires JavaScript, it will stop the crawl
#             if ("You need to enable JavaScript to run this app." in text):
#                 print("Unable to parse page "+url+" due to JavaScript being required.")
            
#             # Otherwise, write the text to the file in the text directory
#             f.write(text)
        
#         # Get the hyperlinks from the URL and add them to the queue
#         for link in get_domain_hyperlinks(local_domain, url):
#             if link not in seen:
#                 queue.append(link)
#                 seen.add(link)
    
# crawl(full_url)

# def remove_newlines(serie):
#     serie = serie.str.replace('\n',' ')
#     serie = serie.str.replace('\\n',' ')
#     serie = serie.str.replace('  ',' ')
#     serie = serie.str.replace('  ',' ')
#     return serie

# # Create a list to store the text files
# texts = []

# # Get all the text files in the text directory
# for file in os.listdir("text/"+domain+"/"):
    
#     # Open the file and read the file
#     with open("text/"+domain+"/"+file, "r", encoding="utf-8") as f:
#         text = f.read()
        
#         # Omit the first 11 lines and last 4 lines, then replace -, _, and #update with spaces
#         texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
# # Create a dataframe from the list of texts
# df = pd.DataFrame(texts, columns=["fname","text"])

# # Set the text column to be the raw text with the newlines removed
# df['text'] = df.fname +". " + remove_newlines(df.text)
# df.to_csv('processed/scrapped.csv')

# Load the cl100K_base tokenizer which is designed to work with ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
df = pd.read_csv('processed/scrapped.csv', index_col=0)
df.columns = ['title', 'text']

# print(df.head())

df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

print(df.head())

df.n_tokens.hist()

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split(". ")
    
    # Get number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" "+sentence)) for sentence in sentences]

    chunks = []
    token_so_far = 0
    chunk = []
    
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        
        # If the number of tokens so far plus the number of tokens in the current sentences is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset the chunk and tokens so far.
        if token_so_far + token > max_tokens:
            chunks.append(". ".join(chunk)+".")
            chunk = []
            token_so_far=0
        
        # If the number of tokens in the current sentence is greater than the max number of tokens, go to the next sentence
        if token > max_tokens:
            continue
        
        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        token_so_far += token +1
        
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk)+".")
    
    return chunks

shortened = []

# Loop through the dataframe
for row in df.iterrows():
    
    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue
    
    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])


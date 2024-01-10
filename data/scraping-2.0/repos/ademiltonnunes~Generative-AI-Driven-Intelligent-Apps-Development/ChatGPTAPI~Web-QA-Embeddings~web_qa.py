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
from ast import literal_eval

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag:str, attrs) -> None:
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

class HyperLink():
    def __init__(self, http_url_pattern :str= r'^http[s]{0,1}://.+$') -> None:
        self.__HTTP_URL_PATTERN = http_url_pattern

    # Function to get the hyperlinks from a URL
    def get_hyperlinks(self, url):

        # Try to open the URL and read the HTML
        try:
            # Open the URL and read the HTML
            with urllib.request.urlopen(url) as response:

                # If the response is not HTML, return an empty list
                if not response.info().get('Content-Type').startswith("text/html"):
                    return []

                # Decode the HTML
                html = response.read().decode('utf-8')
        except Exception as e:
            print(e)
            return []

        # Create the HTML Parser and then Parse the HTML to get hyperlinks
        parser = HyperlinkParser()
        parser.feed(html)

        return parser.hyperlinks

    # Function to get the hyperlinks from a URL that are within the same domain
    def get_domain_hyperlinks(self, local_domain, url):
        clean_links = []
        for link in set(self.get_hyperlinks(url)):
            clean_link = None

            # If the link is a URL, check if it is within the same domain
            if re.search(self.__HTTP_URL_PATTERN, link):
                # Parse the URL and check if the domain is the same
                url_obj = urlparse(link)
                if url_obj.netloc == local_domain:
                    clean_link = link

            # If the link is not a URL, check if it is a relative link
            else:
                if link.startswith("/"):
                    link = link[1:]
                elif (
                    link.startswith("#")
                    or link.startswith("mailto:")
                    or link.startswith("tel:")
                ): 
                    continue
                clean_link = "https://" + local_domain + "/" + link

            if clean_link is not None:
                if clean_link.endswith("/"):
                    clean_link = clean_link[:-1]
                clean_links.append(clean_link)

        # Return the list of hyperlinks that are within the same domain
        return list(set(clean_links))

class Crawl():

    def crawling(self, url):
        # Parse the URL and get the domain
        local_domain = urlparse(url).netloc

        # Create a queue to store the URLs to crawl
        queue = deque([url])

        # Create a set to store the URLs that have already been seen (no duplicates)
        seen = set([url])

        # Create a directory to store the text files
        if not os.path.exists("text/"):
                os.mkdir("text/")

        if not os.path.exists("text/"+local_domain+"/"):
                os.mkdir("text/" + local_domain + "/")

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
                os.mkdir("processed")

        # While the queue is not empty, continue crawling
        while queue:

            # Get the next URL from the queue
            url = queue.pop()
            print(url) # for debugging and to see the progress

            try:
                # Save text from the url to a <url>.txt file
                with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

                    # Get the text from the URL using BeautifulSoup
                    soup = BeautifulSoup(requests.get(url).text, "html.parser")

                    # Get the text but remove the tags
                    text = soup.get_text()

                    # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                    if ("You need to enable JavaScript to run this app." in text):
                        print("Unable to parse page " + url + " due to JavaScript being required")

                    # Otherwise, write the text to the file in the text directory
                    f.write(text)
            except Exception as e:
                print("Unable to parse page " + url)

            # Get the hyperlinks from the URL and add them to the queue
            for link in HyperLink().get_domain_hyperlinks(local_domain, url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)

class Embed():
    def __init__(self, api_key, domain, fullurl, max_tokens = 500) -> None:
        self.__api = openai
        self.__api_key = self.__api.api_key = api_key
        self.__maxtokens = max_tokens
        self.__domain = domain
        self.__fullurl = fullurl
        # Create a list to store the text files
        self.__texts=[]
        self.__shortened = []
        # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
        self.__tokenizer = tiktoken.get_encoding("cl100k_base")
        self.__crawl = Crawl()

    def __remove_newlines(self, serie):
        serie = serie.str.replace('\n', ' ')
        serie = serie.str.replace('\\n', ' ')
        serie = serie.str.replace('  ', ' ')
        serie = serie.str.replace('  ', ' ')
        return serie

    # Function to split the text into chunks of a maximum number of tokens       
    def __split_into_many(self, text, max_tokens = 500):        

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(self.__tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1
            # Add the last chunk to the list of chunks
        if chunk:
            chunks.append(". ".join(chunk) + ".")

        return chunks
    
    def embedding(self):

        #crawling
        self.__crawl.crawling(self.__fullurl)

        # Get all the text files in the text directory
        for file in os.listdir("text/" + self.__domain + "/"):

            # Open the file and read the text    
            with open("text/" + self.__domain + "/" + file, "r", encoding="UTF-8") as f:    
                text = f.read()

                # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
                self.__texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

        # Create a dataframe from the list of texts
        df = pd.DataFrame(self.__texts, columns = ['fname', 'text'])

        # Set the text column to be the raw text with the newlines removed
        df['text'] = df.fname + ". " + self.__remove_newlines(df.text)
        df.to_csv('processed/scraped.csv', escapechar='\\', index=True)  # Change '\\' to the escape character you prefer
        
        #Read csv file
        df = pd.read_csv('processed/scraped.csv', index_col=0)
        df.columns = ['title', 'text']

        # Tokenize the text and save the number of tokens to a new column
        df['n_tokens'] = df.text.apply(lambda x: len(self.__tokenizer.encode(x)))        
      
        # Loop through the dataframe
        for row in df.iterrows():

            # If the text is None, go to the next row
            if row[1]['text'] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]['n_tokens'] > self.__maxtokens:
                self.__shortened += self.__split_into_many(row[1]['text'], self.__maxtokens)

            # Otherwise, add the text to the list of shortened texts
            else:
                self.__shortened.append( row[1]['text'] )

        df = pd.DataFrame(self.__shortened, columns = ['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(self.__tokenizer.encode(x)))

        df['embeddings'] = df.text.apply(lambda x: self.__api.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
        df.to_csv('processed/embeddings.csv')
        
        df['embeddings'] = df.text.apply(lambda x: self.__api.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

        df.to_csv('processed/embeddings.csv')

class Ask():
    def __init__(self, api_key:str) -> None:
        self.__df = pd.read_csv('processed/embeddings.csv', index_col=0)
        self.__api = openai
        self.__api_key = self.__api.api_key = api_key
        
        
    def __create_context(self,question, df, max_len=1800, size="ada"):
        # Get the embeddings for the question
        q_embeddings = self.__api.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def __answer_question(
        self,
        df,
        model="gpt-3.5-turbo-instruct",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
    ):
        
        context = self.__create_context(question,df,max_len=max_len,size=size,)
        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        try:
            # Create a completions using the question and context
            response = self.__api.Completion.create(
                prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=model,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(e)
            return ""

    def answerQuestion(self, question:str):
        self.__df['embeddings'] = self.__df['embeddings'].apply(literal_eval).apply(np.array)
        return self.__answer_question(self.__df, question= question)

def main() -> None:

    #Openning .env file with api key
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    # Define API Key
    api_key:str = str(os.environ.get("OPENAI_API_KEY"))
    
    # Define root domain to Embedding
    domain = "sfbu.edu"
    full_url = "https://sfbu.edu/"

    #Embed a website
    Embed(api_key, domain,full_url).embedding()

    # #Asking Questions 
    # question = "What is SFBU?"
    # answer = Ask(api_key).answerQuestion(question)
    # print(question)
    # print(answer)
    # print()

    # question = "What day is it?"
    # answer = Ask(api_key).answerQuestion(question)
    # print(question)
    # print(answer)
    # print()



if __name__== "__main__":
    main()
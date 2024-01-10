################################################# SETUP #################################################

# Import libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
# import PyPDF2
# from io import BytesIO
import openai
import os
from urllib.parse import urlparse
from requests.exceptions import RequestException
import configparser
import json
from requests_html import HTMLSession

# config API keys
config = configparser.ConfigParser()
config.read('config.ini')
openai_key = config['openai']['key']
openai.api_key = openai_key

################################################# SETUP #################################################

# SETTING UP DATAFRAME
# List of example URLs on the GSL water crisis as .txt
links_txt = "data/links.txt"

# filepath for JSON links on the GSL water crisis
links_json = "data/links.json"

# save filepath of keywords on the GSL water crisis
keywords_fp = "data/keywords.txt"

# Initialize an empty DataFrame to store the data
"""
DataFrame Columns:
    - source: The origin of the response as a name/type (e.g., 'website', 'youtube', 'reddit', etc.).
    - filepath: The filepath to the text content of the response. This is the main body of the response.
    - content: The text content of the response, cleaned up and ready for analysis. 
    - url: The URL or source link of the original response. This provides a reference to the original content.
    - author_names: A list of the authors involved in creating the response. This could be the username of a Reddit or Twitter user, the name of a YouTube channel, or the author of a news article or report.
    - value_types: A list of the major value types in the response. This represents the main themes or values that the response is promoting or discussing.
    - stakeholder_types: A list of the types of stakeholders in the response. This represents the groups or individuals who have a stake in the Great Salt Lake crisis, as identified in the response.
    - keywords: A list of the top 5 significant non-stop-words used in the response. These are the words that are most relevant to the content of the response, excluding common stop words like 'the', 'and', 'is', etc.
    - methods: A list of the research methods, techniques, or ways of analyzing the problem that are mentioned in the response. This could include scientific research methods, policy analysis techniques, or other methods of understanding and addressing the crisis.
    - solutions: A list of the solutions to the crisis proposed in the response. These are the specific actions or strategies suggested to address the Great Salt Lake crisis.
    - facts: A list of the facts, numbers, results, or takeaways in the response. This includes any specific data or factual information presented in the response, such as the cost of a proposed solution or the amount of water it could save.
"""

blank_df = pd.DataFrame(columns=["source", "filepath", "content", "url", "author_names",
                           "value_types", "stakeholder_types", "keywords",
                           "methods", "solutions", "facts"])

################################################# DATA COLLECTION #################################################
# Scraping content from URLs in the .txt file

# HELPER FUNCTIONS
def get_source(url):
    """
    Function to extract base URL (source) from a URL.

    Parameters:
    url (str): The URL from which you want to extract base URL.

    Returns:
    str : Base URL extracted from given URL.
    """
    return urlparse(url).netloc

def get_author(soup):
    """
    Function to extract author from a BeautifulSoup object.

    Parameters:
    soup (bs4.BeautifulSoup): BeautifulSoup object containing HTML code.

    Returns:
    str : Author name extracted from given HTML code.

    """
    # Try to find a meta tag with name="author"
    author_tag = soup.find("meta", attrs={"name": "author"})
    if author_tag:
        return author_tag["content"]

    # Try to find a span, p, or div tag with class="author"
    for tag_name in ["span", "p", "div"]:
        author_tag = soup.find(tag_name, class_="author")
        if author_tag:
            return author_tag.get_text(strip=True)

    # Try to find the author in the URL (e.g., "www.example.com/author-name/article-title")
    url_path = urlparse(soup.url).path
    url_path = url_path.decode('utf-8') # decode the url path
    for segment in url_path.split("/"):
        if "author" in segment.lower():
            return segment

    # If all else fails, return "Unknown"
    return "Unknown"

# def read_pdf_from_url(url):
#     """
#     Read a PDF from a URL and return its content as a string.

#     Parameters:
#     url (str): The URL of the PDF.

#     Returns:
#     str: The content of the PDF.
#     """
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception if the response status is not 200 (OK)
#     except requests.RequestException as e:
#         print(f"Failed to fetch {url} due to error: {e}")
#         return None  # Return None if the request failed

#     # Open the PDF
#     with BytesIO(response.content) as open_pdf_file:
#         read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
#         text = ""
#         for page in range(read_pdf.getNumPages()):
#             text += read_pdf.getPage(page).extractText()

#     return text

# MAIN SCRAPING FUNCTION
def scrape_content(df, json_filepath):
    """
    Scrape content from a list of URLs and save the content to .txt files.

    Parameters:
    df (pd.DataFrame): An existing DataFrame to append the scraped data to.
    json_filepath (str): The path to a JSON file containing the URLs to scrape.

    Returns:
    pd.DataFrame: The DataFrame with the appended scraped data.
    """
    # Load the data from the JSON file
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Scrape content from URLs
    for source, urls in data.items():
        for url in urls:
            # # skip research articles for now 
            # if source == "research":
            #     continue
            
            # Skip if the URL is not valid
            if not url or not urlparse(url).scheme:
                continue
            
            # # Check if the URL is a link to a PDF
            # if url.lower().endswith('.pdf'):
            #     content = read_pdf_from_url(url)
            #     author = None  # We can't get the author from a PDF=
            # Otherwise, assume it's a link to an HTML web page
            try:
                headers = { # add headers to trick the browser
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                        }
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise an exception if the response status is not 200 (OK)
            except requests.RequestException as e:
                print(f"Failed to fetch {url} due to error: {e}")
                continue  # Skip to the next URL

            soup = BeautifulSoup(response.text, 'html.parser')

            # Get the author
            author = get_author(soup)

            # Get the content
            # Find all tags that contain relevant content
            tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            content = "\n".join([i.get_text() for i in soup.find_all(tags)])
            
            # If the source is Reddit, also scrape comments from the old Reddit layout
            if "reddit" in url:
                # Use requests-html instead of requests and BeautifulSoup
                # This allows you to render JavaScript on the page and access dynamic content
                # See https://requests-html.kennethreitz.org/ for details
                session = HTMLSession()
                old_reddit_url = url.replace("https://www.reddit.com", "https://old.reddit.com")
                response = session.get(old_reddit_url)
                # Render the page for 5 seconds to load the comments
                response.html.render(sleep=5)
                # Find all the comment elements by their CSS selector
                comments = response.html.find('.usertext-body')
                content += "\n\n".join([comment.text for comment in comments])
                content += "\n".join([i.get_text() for i in soup.find_all('div')]) # add div tags as well
                
            # Check if content is empty
            if not content.strip():
                print(f"Content empty - Scraping failed for {url}")
                continue

            # Create a unique filename for each URL
            parsed_url = urlparse(url)
            website_name = parsed_url.netloc.split('.')[0] # get the website name
            article_name = parsed_url.path.strip("/").replace("/", "_")
            filename = f"responses/{source}/{website_name}_{article_name}.txt"
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Save the content to a .txt file
            try:
                with open(filename, "w") as file:
                    file.write(content)
            except IOError as e:
                print(f"Failed to write to file {filename} due to error: {e}")
                continue  # Skip to the next URL

            # Add the data to the DataFrame
            try:
                df = df.append({
                    "source": source,
                    "filepath": filename,
                    "url": url,
                    "author_names": author,
                    # The rest of the columns will be filled in later
                    "value_types": None,
                    "stakeholder_types": None,
                    "keywords": None,
                    "methods": None,
                    "solutions": None,
                    "facts": None
                }, ignore_index=True)
            except Exception as e:
                print(f"Failed to add data to DataFrame due to error: {e}")
            
    return df
    
# execute the function
responses_df = scrape_content(blank_df, links_json)

# Save resulting DataFrame to a CSV file
output_filepath = "data/responses.csv"
responses_df.to_csv(output_filepath, index=False)

# ################################################# DATA CLEANING #################################################
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

def clean_text_file(row):
    """
    Function to clean text from a file removing stopwords and punctuation.
    
    Parameters:
    row (pd.Series): DataFrame row.
    
    Returns:
    str: Cleaned string of text.
    """
    # Read the text data from the file
    with open(row['filepath'], 'r') as file:
        text = file.read()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert to lower case
    tokens = [word.lower() for word in tokens]
    
    # Remove punctuation and non-alphabetic tokens
    words = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Finalize the text content 
    text = ' '.join(words)

    # Count the frequency of each word
    word_freq = nltk.FreqDist(words)

    # Get the top 5 most frequent words
    top_keywords = [word for word, freq in word_freq.most_common(5)]

    return (text, top_keywords)

responses_df[['content', 'keywords']] = responses_df.apply(clean_text_file, 
                                                           axis=1, result_type='expand')

responses_df.to_csv("data/cleaned_responses.csv", index=False)


# ################################################# AI STUFF ################################################# 


# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt="Translate the following English text to French: '{}'",
#   max_tokens=60
# )
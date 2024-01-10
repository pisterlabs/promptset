from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
import tiktoken
import time
import os
import csv
import openai
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# models
GPT_MODEL = "gpt-3.5-turbo"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text_chunks(text: str, max_tokens: int):
    """Split a text into chunks of specified maximum tokens."""
    chunks = []
    current_chunk = ""

    for word in text.split():
        current_chunk += word + " "

        if num_tokens(current_chunk) >= max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
openai.api_key = OPENAI_API_KEY

# Load the list of URLs
with open('./additional.txt', 'r') as file:
    urls = [line.strip() for line in file.readlines()]

# Set up the driver options
options = webdriver.ChromeOptions()
options.add_argument("--incognito")
options.add_argument("--disable-site-isolation-trials")
options.add_argument("--headless")

# Create a new instance of the Chrome driver
driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))

# Maximum tokens allowed for the model
max_tokens = 8190

embedding_file_name = 'visionos_docs_2023_07_10_embedding.csv'
text_file_name = 'visionos_docs_2023_07_10_text.csv'

# Open the CSV file for saving embedding
with open(embedding_file_name, 'w', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Write column names to the CSV file
    writer.writerow(['id', 'embedding'])

# Open the CSV file for saving text
with open(text_file_name, 'w', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Write column names to the CSV file
    writer.writerow(['id', 'text'])

# For each URL
for idx, url in enumerate(urls):
    # Navigate to the page
    driver.get(url)

    # Wait for the page to load
    time.sleep(5)

    # Get the HTML of the page
    html_source = driver.page_source
    # print(html_source)

    # Remove 'nav' tags
    driver.execute_script("var elems = document.getElementsByTagName('nav'); for (var i = 0; i < elems.length; i++) { elems[i].style.display = 'none'; }")

    # Remove all content within 'footer' tags
    driver.execute_script("var footers = document.getElementsByTagName('footer'); for (var i = 0; i < footers.length; i++) { while (footers[i].firstChild) { footers[i].removeChild(footers[i].firstChild); } }")

    # Initialize an empty string to hold all the text
    text = ""
    
    # Find all the h1, h2, h3, h4, p, ul, and code tags and append their text to the string
    for tag_name in ['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'code']:
        elements = driver.find_elements(By.TAG_NAME, tag_name)
        # For each element, append its text to the string immediately
        for element in elements:
            try:
                text += element.text + " "
            except StaleElementReferenceException:
                continue

    text_chunks = split_text_chunks(text, max_tokens)

    print("chunks: ", len(text_chunks))
    print(text_chunks)

    embeddings = []
    # Iterate over text chunks and retrieve embeddings
    for chunk in text_chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    embedding = np.array(embeddings).mean(axis=0).tolist()
    

    print("embedding: ", len(embedding))
    # print(embedding)
    print(f"Total tokens in the text: {num_tokens(text)}")

    # Now you can use the 'text' string with the OpenAI API
    # print(text)

    # Open the CSV file for saving embedding
    with open(embedding_file_name, 'a', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        
        # Write embedding vector to the CSV file
        writer.writerow([idx, ",".join(map(str, embedding))])

    # Open the CSV file for saving text
    with open(text_file_name, 'a', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write text to the CSV file
        writer.writerow([idx, text])

# Close the driver
driver.quit()
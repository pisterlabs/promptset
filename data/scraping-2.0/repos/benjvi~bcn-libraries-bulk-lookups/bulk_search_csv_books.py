import csv
from bs4 import BeautifulSoup
from dataclasses import dataclass
import requests
import time
from googletrans import Translator
import openai
import os

@dataclass
class Book:
    original_title: str
    translated_title: str
    author: str

books = []
translator = Translator()
with open('goodreads_library_export.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)  # Skip the header row

    # Find the indexes of the csv fields we want, based on header values
    title_index = header.index("Title") 
    author_index = header.index("Author")
    shelf_index = header.index("Exclusive Shelf")

    i = 0
    max_books = 100
    for row in csvreader:
        if row[shelf_index] == 'to-read' and i < max_books:
            i += 1
            # Translate the title to Spanish
            translated_title = translator.translate(row[title_index], dest='es').text
            print(f'Original Title: {row[title_index]}, Translated Title: {translated_title}')

            books.append(Book(row[title_index], translated_title, row[author_index]))

i = 0
for book in books:
    i += 1
    if i < 100:
        print(f'Scraping Title: {book.translated_title}, Author: {book.author}')
        
        # Call library catalog search, by building URL in the following form:
        # https://aladi.diba.cat/search~S1*spi/X?SEARCH=t:(1234)+and+a:(Morrison)&searchscope=171&SORT=AX
        # Where 1234 is the title, and Morrison is the author

        response = requests.get("https://aladi.diba.cat/search~S1*spi/X?SEARCH=t:({})+and+a:({})&searchscope=171&SORT=AX".format(book.translated_title, book.author))
        # site is using rate-limiting, so we wait 1 second between requests
        time.sleep(1)

        # Parse the HTML response
        # Check for a successful response
        if response.status_code == 200:
            empty_results = False

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the specific div by its class
            target_div = soup.find('div', {'class': 'pageContentColumn'})
            
            # Check if the div exists
            if target_div:
                 # Find all 'h2' elements directly underneath the target div
                h2_elements = target_div.find_all_next('h2', limit=2)  # Limit to 2 to find just the first two
                
                # Check if at least two 'h2' elements are found
                if len(h2_elements) >= 2:
                    # Extract the text from the second 'h2' element
                    second_h2_text = h2_elements[1].text.strip()
                    
                    # Check for the particular string indicating empty results
                    # NB that if this element doesn't exist, we assume we have  results
                    if second_h2_text == 'NO HAY RESULTADOS':
                        empty_results = True
                        print(f'Results are empty for book "{book.translated_title}" by "{book.author}"')
                    
            if not empty_results:
                print(f'Found results at URL: {response.url}')
                
        else:
            print(f'Failed to fetch the page for book. HTTP Status Code: {response.status_code}')

# TODO: use this function to translate the book titles
def gpt_translate(sentence, source_lang, target_lang, model = "gpt_4"):#source_lang, target_lang are names of the languages, like "French"
    openai.api_key = os.environ.get("OPENAI_API_KEY") #or supply your API key in a different way
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Please translate the user message from {source_lang} to {target_lang}. Make the translation sound as natural as possible."
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
        temperature=0
    )
    return completion["choices"][0]["message"]["content"]
        
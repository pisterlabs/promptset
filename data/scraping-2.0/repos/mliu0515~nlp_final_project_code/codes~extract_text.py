from bs4 import BeautifulSoup
import requests
import os
from transformers import GPT2Tokenizer
import random
import re
import json
import openai


"""This is the code that I ran to extract the first sentence of each book. I used LLM to help me with that"""

openai.api_key = 'your_key'
csv_file = "all_books.csv"
one_pattern = re.compile(r'Chapter', re.IGNORECASE)

# given a csv file, store the third column in a list
def get_urls_from_csv(csv_file):
    urls = {} 
    # key: first column, value: dictionary {url: third column, author: second column, bias_type: fourth column, description: fifth column}
    with open(csv_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(",")
            urls[line[0]] = {}
            urls[line[0]]["url"] = line[2]
            urls[line[0]]["author"] = line[1]
            urls[line[0]]["bias_type"] = line[3]
            urls[line[0]]["description"] = line[4]
            urls[line[0]]["first_sentence"] = ""
    return urls

def sample_first_sentence(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    split_text = one_pattern.split(soup.prettify())
    longest_chapter = max(split_text, key=len)
    split_longest_chapter = longest_chapter.split("\n\r")
    max_paragraph = max(split_longest_chapter, key=len)
    prompt = "what is the first sentece in this paragraph? Return in just text and nothing else. The paragraph is: \n" + max_paragraph
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate model
        prompt=prompt,
        max_tokens=500  # Adjust as necessary
    )
    first_sentence = response.choices[0].text.strip()
    print(first_sentence)
    return first_sentence


# Do the __main__ thing
if __name__ == "__main__":
    if "book_contents" not in os.listdir():
        os.mkdir("book_contents")

    urls = get_urls_from_csv(csv_file)
    for book_title, info in urls.items():
        print(book_title)
        sentence = sample_first_sentence(info["url"])
        # Update the original csv file by adding a new column where the column vlue is the content
        urls[book_title]["first_sentence"] = sentence
        
    # dump urls in a json file
    with open("all_books_with_prompt_sentence.json", "w") as file:
        json.dump(urls, file, indent=4)

    print("done")
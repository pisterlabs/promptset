#Program to download pdf files from a URLs to a specified directory
#User Input is the url and the directory to store the pdf files
#Maximun number of files to download is 10
#If the file name is longer than 50 characters, it will be ignored

import os
import re
import requests
from bs4 import BeautifulSoup
import openai

url = input("Enter the URL: ")
directory = input("Enter the directory to store the pdf files: ")
max_files = 10

# Set the OpenAI keys for the Completion API call
openai.api_key = os.environ['OPENAI_API_KEY']
openai.engine = 'davinci'


def download_pdf(site, directory, max_files):
    # Make the request to the website
    r = requests.get(site)

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(r.text, 'html.parser')

    # Find all pdfs on the page
    pdfs = soup.findAll('a', href=re.compile(r'([^\s]+(\.(?i)(pdf))$)'))

    # Using a regular expression to look at a variable and find all email addresses
    email = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')

    # API call to OpenAI to extract email addresses from the text
    response = openai.Completion.create(
        engine="davinci",
        prompt=email,
        max_tokens=5,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created: {}".format(directory))

    # Download all pdfs
    count = 0
    for pdf in pdfs:
        if count == max_files:
            break
        if len(pdf.text) > 50:
            print("File name too long: {}".format(pdf.text))
            continue
        with open(os.path.join(directory, pdf.text), 'wb') as f:
            if 'http' not in pdf['href']:
                # Sometimes a pdf source can be relative
                # If it is, provide the base url which also happens
                # to be the site variable at the moment.
                url = '{}{}'.format(site, pdf['href'])
            else:
                url = pdf['href']
            response = requests.get(url)
            f.write(response.content)
            count += 1

# Call the function with the URL of the website you want to download pdfs from
download_pdf(url, directory, max_files)

# Path: slides/src/downloadimages.py


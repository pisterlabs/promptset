import json
import openai
from openai.embeddings_utils import cosine_similarity
import matplotlib
from langchain.embeddings import OpenAIEmbeddings
from llama_index import download_loader
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from urllib.request import urlopen
from bs4 import BeautifulSoup

from langchain.document_loaders import PyPDFLoader

import csv
embeddings = OpenAIEmbeddings(
    openai_api_key="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")
openai.api_key = "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"


def getCSV(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        csv_string = ""
        column_headers = []
        for row in csv_reader:
            if line_count == 0:
                for col in row:
                    column_headers.append(col)
                    csv_string += f'"{col}": "{col}",'
                line_count += 1
            else:
                csv_string += "{"
                i = 0
                for col in row:
                    csv_string += f'"{column_headers[i]}": "{col}",'
                    i += 1
                csv_string += "},"
                line_count += 1
        return csv_string



def getJSON(filename):
    # turn the json file into a json string
    json_string = ""
    json_file = open(filename)
    json_data = json.load(json_file)
    for item in json_data:
        json_string += json.dumps(item)
    return json_string


def getWebsite(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def getPDF(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    return pages

 
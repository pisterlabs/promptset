from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, UnstructuredURLLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

import PyPDF2
import torch
import requests
import mechanicalsoup
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import os
import csv


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DB_FAISS_PATH = "vectorstores/db_faiss"
"""
Fetches all links within a given web page and appends them to the provided sets based on certain criteria.

Parameters:
base_url (str): Base URL of the page
url (str): URL of the page to fetch links from
setOfInsideLinks (set): Set to store valid internal links
setOfWrongLinks (set): Set to store invalid or broken links
browser (mechanicalsoup.StatefulBrowser): Browser object for making HTTP requests
headers (dict): HTTP headers for making requests
level (int): Current level of depth in fetching links

Returns:
None
"""
def getAllLinksInPage(base_url, url, setOfInsideLinks, setOfWrongLinks, browser, headers, level):
    # Define the maximum level of page traversal
    max_level = 1
    delay = 2
    time.sleep(delay)

    try:
        # Fetch the webpage content from the specified URL using MechanicalSoup
        page = browser.get(url, headers=headers, timeout=5)
        
        # Check if the page or its content is not retrievable
        if page == None or page.soup == None:
            setOfWrongLinks.add(url)
            return 
        
        if page.status_code == 404:
            setOfWrongLinks.add(url)  
            print(f"404 Not Found: {url}")  
            return  
    except Exception as e:
        print(url) 
        print(f"{e}")  
        setOfWrongLinks.add(url) 
        return 

    time.sleep(delay) 

    # Find all anchor and link elements on the page and gather their 'href' attributes
    links = page.soup.find_all('a')
    links += page.soup.find_all('link')

    # Iterate through all found links
    for link in links:
        href = link.get('href') 
        
        # Format the URL link if necessary
        if href and href[-1] == "/":
            href = href[0:len(href)-1]

        # Filter out specific types of URLs based on certain conditions
        if href and "http" in href:
            continue
        elif href and (base_url + href).rfind("html") == (base_url + href).find("html") and \
        href.rfind("pdf") == -1 and href.rfind("png") == -1 and href.rfind("json") == -1 and href.rfind(":") == -1 and \
        href.rfind(".ico") == -1 and href.rfind(".svg") == -1 and href.rfind(".si") == -1 and href.rfind("?") == -1 and \
        href.rfind("%20") == -1 and href.rfind("#") == -1 and (base_url + href).rfind(".com") == (base_url + href).find(".com"):

            link = ""

            # Construct the absolute link from the base URL and extracted href
            if href[0] != "/" and base_url[-1] != "/":
                link = base_url + "/" + href
            elif href[0] == "/" and base_url[-1] == "/":
                link = base_url + href[1:]
            else:
                link = base_url + href

            if link in setOfWrongLinks or link in setOfInsideLinks:
                continue
            
            # If the current traversal level is less than the maximum level, continue extracting links recursively
            if level < max_level:
                getAllLinksInPage(base_url, link, setOfInsideLinks, setOfWrongLinks, browser, headers, level + 1)
            setOfInsideLinks.add(link)
            print("URL:", link) 



def listOfCenters(browser, headers):
    delay = 2
    time.sleep(delay)

    # Fetch the webpage content from the specified URL using MechanicalSoup
    page = browser.get('https://www.nysdra.org/centers', headers=headers, timeout=5)
    time.sleep(delay)

    listOfCenters = set()
    documentList = []

    # Find all anchor elements on the webpage
    links = page.soup.find_all('a')

    # Iterate through all found links
    for link in links:
        # Get the 'href' attribute from each link
        href = link.get('href')
        
        # Check if the link points to a PDF file
        if href and ".pdf" in href:
            # Download the PDF content
            response = requests.get(href)
            with open("temp.pdf", "wb") as f:
                f.write(response.content)

            # Read and extract text from the downloaded PDF
            pdf_file = open("temp.pdf", "rb")
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for num in range(len(reader.pages)):
                page = reader.pages[num]
                text += page.extract_text()

            # Close and remove the temporary PDF file
            pdf_file.close()
            os.remove("temp.pdf")

            # Append the extracted text as a Document object to the document list
            documentList.append(Document(page_content=text.replace("\n", "").replace("\x00", "f"), metadata={"source": href}))

        # Check for other types of links excluding certain domains and resources
        elif href and "http" in href and href.lower().find("nysdra") == -1 \
        and href.lower().find("youtube") == -1 and href.lower().find("linkedin") == -1 and href.lower().find("map") == -1:
            # Initialize sets for inside and wrong links
            setOfInsideLinks = set()
            setOfWrongLinks = set()
            setOfInsideLinks.add(href)
            
            # Fetch all links within the current link recursively using a helper function
            getAllLinksInPage(href, href, setOfInsideLinks, setOfWrongLinks, browser, headers, 0)
            
            # Union of unique inside links with the overall set of centers
            listOfCenters = listOfCenters.union(setOfInsideLinks)

    # Return the list of unique center links and extracted documents
    return (list(listOfCenters), documentList)


def createVectorDB():
    # Define the user-agent header for the browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Create a browser object using MechanicalSoup
    browser = mechanicalsoup.StatefulBrowser()

    # Fetch information on centers from the specified website
    infoTuple = listOfCenters(browser, headers)

    # Extract URLs and PDF documents from the fetched information tuple
    URLs = infoTuple[0]
    pdfDocumentList = infoTuple[1]

    # Display the extracted URLs
    print(URLs)

    # Load unstructured data from URLs using a specific set of headers
    loaders = UnstructuredURLLoader(urls=URLs, headers=headers)
    documents = loaders.load()

    # Combine loaded documents with PDF documents
    documents += pdfDocumentList

    # Load text data from a text file into documents
    loaders3 = TextLoader(file_path="data/QA.txt", encoding="utf-8")
    documents3 = loaders3.load()
    documents += documents3

    # Replace newline characters with empty strings in all document content
    for document in documents:
        document.page_content = document.page_content.replace("\n", "")

    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings using a pre-trained HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": DEVICE})

    # Create a FAISS database from the document texts and embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS database locally
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    createVectorDB()
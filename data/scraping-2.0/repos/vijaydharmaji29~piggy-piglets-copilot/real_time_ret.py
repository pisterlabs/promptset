#retrieve links and add to db

import requests
from bs4 import BeautifulSoup
import markdownify

import os
import sys
import time

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

db_name_global = ""

# Import necessary modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.memory import ConversationBufferMemory

flag = 0
links_visited = set()

def add_to_vdb(db_name, content_md):
    # Open and read the Markdown file
    # with open("./docs/" + file_path, "r", encoding="utf-8") as md_file:
    #     markdown_content = md_file.read()

    markdown_content = content_md

    markdown_document = markdown_content
    #documents needs to be text spliter

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Char-level splits
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    documents = splits

    if len(documents) != 0:
        

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)

        # Define the directory where you want to save the persisted database
        persist_directory = db_name

        # Initialize OpenAIEmbeddings for embedding
        embedding = OpenAIEmbeddings()

        # Create and persist the Chroma vector database
        vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

        # Persist the database to disk
        vectordb.persist()
    else:
        print("UNECESSARY PAGE")

def remove_css_from_html(input_html):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(input_html, 'html.parser')

    # Remove all <style> tags (CSS)
    for style_tag in soup.find_all('style'):
        style_tag.extract()

    # Return the modified HTML
    return str(soup)

def remove_javascript_from_html(input_html):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(input_html, 'html.parser')

    # Remove all <script> tags
    for script_tag in soup.find_all('script'):
        script_tag.extract()

    # Remove inline event handlers (e.g., onclick="...")
    for tag in soup.find_all():
        for attr in list(tag.attrs.keys()):
            if attr.startswith("on"):
                del tag.attrs[attr]

    # Return the modified HTML
    return str(soup)


def get_inside(website):
    global flag
    result = requests.get(website)
    content = result.text
    soup = BeautifulSoup(content, 'lxml')

    box = soup.find_all('html')    

    if len(box) < 1:
        return "", []
    
    links = set()
    for b in box:
        for link in b.find_all('a', href=True):
            links.add(link['href'])

    best_html_str = remove_javascript_from_html(str(soup.html))
    best_html_str = remove_css_from_html(best_html_str)

    md = markdownify.markdownify(best_html_str, heading_style="ATX")
    flag += 1
    return (md, links)
    
    # return (remove_javascript_from_html(soup.html), links)

def get_page_content(website, round, db_name):
    global flag
    global db_name_global
    db_name_global = db_name
    content_md, links = get_inside(website)
    folder = "moveworks/"
    

    file_name = website
    file_name = file_name.replace("https://www.", "")
    file_name = file_name.replace("https://", "")
    file_name = file_name.replace("http://", "")
    file_name = file_name.replace(".", "(dot)")
    file_name = file_name.replace("/", "_")
    file_name = file_name + ".md"
    file_name = folder + file_name

    
    
    f = open(file_name, "w")
    f.write(content_md)

    add_to_vdb(db_name_global, content_md)

    f.close()

    

    if flag < 100:
        print(round)
        for l in links:
            # print(l)
            if "www" not in l:
                l = "www." + l
            try:
                get_page_content(l, round + 1)
            except:
                # get_page_content(str(website) + l, round + 1)
                # print("ERROR")
                pass


def real_time_additon(website):
    website = website

    name = website.replace("https", "")
    name = name.replace("http", "")
    name = name.replace("://", "")
    name = name.replace("www.", "")
    name = name.split(".")[0]
    name = "db_" + name

    db_name_global = name

    get_page_content(website, 0, db_name_global)




if __name__ == "__main__":
    # website = "https://www.moveworks.com/"
    # website = "https://subslikescript.com/movies"
    website = "https://www.moveworks.com/"

    if len(sys.argv) > 0:
        website = sys.argv[1]

    print("running for website")

    name = website.replace("https", "")
    name = name.replace("http", "")
    name = name.replace("://", "")
    name = name.replace("www.", "")
    name = name.split(".")[0]
    name = "db_" + name

    print(name)

    db_name_global = name

    get_page_content(website, 0, db_name_global)
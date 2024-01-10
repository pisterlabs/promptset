#!/usr/bin/env python

import argparse
import base64
import configparser
import fnmatch
import os
import requests

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


GITHUB_TOKEN = "[INSERT YOUR GITHUB ACCESS TOKEN HERE]"

def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

def get_files_from_github_repo(owner, repo, branch, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")

def fetch_file_contents(files, ext):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], f"*.{ext}"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents

def get_source_chunks(files, ext):
    print("In get_source_chunks ...")
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_file_contents(files, ext):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks

def main():
    parser = argparse.ArgumentParser(description="Fetch all files from a GitHub repository.")
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument("--branch", default="master", help='Branch to use')
    parser.add_argument('--ext', required=True, help='File extension for files to fetch')
    parser.add_argument('--conf', default='.query.conf', help='Configuration file to use')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read_file(open(args.conf))
    GITHUB_TOKEN = config.get('github', 'token')
    os.environ['OPENAI_API_KEY'] = config.get('openai', 'api_key')

    GITHUB_OWNER, GITHUB_REPO = parse_github_url(args.url)
    
    all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, args.branch, GITHUB_TOKEN)

    CHROMA_DB_PATH = f'./chroma/{os.path.basename(GITHUB_REPO)}'

    chroma_db = None

    if not os.path.exists(CHROMA_DB_PATH):
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(all_files, args.ext)
        chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        chroma_db.persist()
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())

    while True:
        print('\n\n\033[31m' + 'Ask a question' + '\033[m')
        user_input = input()
        print('\033[31m' + qa.run(user_input) + '\033[m')

if __name__ == "__main__":
    main()

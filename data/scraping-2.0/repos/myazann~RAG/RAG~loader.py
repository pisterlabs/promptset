import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
from git import Repo
import os

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, TelegramChatFileLoader, SeleniumURLLoader, GitLoader
from langchain.utilities import SQLDatabase

class FileLoader():

    def load(self, file_name, pdf_loader="unstructured"):

        file_type = self.get_file_type(file_name)
        if file_type == "db":
            print("Reading database!")
            doc = SQLDatabase.from_uri(f"sqlite:///{file_name}") 
        elif file_type == "csv":
            doc = pd.read_csv(file_name)
        else:
            if "telegram" in file_name:
                loader = TelegramChatFileLoader(file_name)
            elif file_type == "url":
                all_urls = set()
                all_urls.add(file_name)
                parsed = urlparse(file_name)
                base_url_path = f"{parsed.scheme}://{parsed.netloc}"
                l1_all_urls = self.get_all_links(file_name, base_url_path)
                all_urls.update(l1_all_urls)
                for url in l1_all_urls:
                    l2_urls = self.get_all_links(url, base_url_path)
                    all_urls.update(l2_urls)
                all_urls = list(all_urls)
                loader = SeleniumURLLoader(urls=all_urls)
            elif file_type == "git":
                repo_name = "/".join(file_name.split("/")[-2:])
                repo_path = f"./files/git/{repo_name}"
                if os.path.exists(repo_path):
                    r = Repo(repo_path)
                else:
                    r = Repo.clone_from(file_name, repo_path)
                loader = GitLoader(repo_path=repo_path, branch=r.heads[0])
            elif file_type == "pdf":
                loader = self.pdf_loaders()[pdf_loader](file_name)
            elif file_type == "txt":
                loader = TextLoader(file_name)
            doc = loader.load()

        return doc

    def get_file_type(self, file_name):
        if file_name.startswith("http"):
            if "github.com" in file_name:
                return "git"
            else:
                return "url"
        else:
            return file_name.split(".")[-1]

    def pdf_loaders(self):
        return {
            "structured": PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
        }

    def get_all_links(self, url, base_url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        urls = set()
        for a in soup.select("a"):
            href = a.get("href")
            if base_url in href:
                urls.add(href)
            elif href.startswith("/"):
                abs_ref = f"{base_url}{href}"
                urls.add(abs_ref)
        return list(urls)
    
    def remove_empty_space(self, doc):
        for page in doc:    
            page.page_content = re.sub(r'\n+', '\n', page.page_content) 
            page.page_content = re.sub(r'\s{2,}', ' ', page.page_content)
        return doc
import requests
import xml.etree.ElementTree as ET
import json
import datetime
from typing import List, Optional
import os
from data_models import Article
from langchain.document_loaders import PyPDFLoader


class arxivApi:
    def __init__(self):
        self.baseUrl = "http://export.arxiv.org/api/query"


    def search_category(self, category: str, num_results: int = 10,
                        minDate: Optional[str] = None,
                        maxDate: Optional[str] = None) -> List[Article]:
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": num_results
        }
        response = requests.get(self.baseUrl, params=params)
        # get the XML
        root = ET.fromstring(response.content)
        # create output dict
        resSet = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            id_ = entry.find('{http://www.w3.org/2005/Atom}id').text
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            categories = [category.get('term') for category in entry.findall('{http://www.w3.org/2005/Atom}category')]
            publishDate = entry.find('{http://www.w3.org/2005/Atom}published').text
            links = [link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link')]
            
            article = Article(
                title=title,
                summary=summary,
                categories=categories,
                publishDate=publishDate,
                link=[l for l in links if 'pdf' in l][0]
            )
            # check if published date is in range
            if minDate and maxDate:
                if minDate <= datetime.datetime.strptime(publishDate, "%Y-%m-%dT%H:%M:%SZ") <= maxDate:
                    resSet.append(article)
            else:
                resSet.append(article)
        return resSet
    

    def search_keywords(self, keywords: str, num_results: int = 200,
                        minDate: Optional[str] = None,
                        maxDate: Optional[str] = None) -> List[Article]:
        params = {
            "search_query": keywords,
            "sortBy": "relevance",
            "sortOrder": "descending",
            "max_results": num_results,
            "start": 0
        }
        response = requests.get(self.baseUrl, params=params)
        # get the XML
        root = ET.fromstring(response.content)
        # create output dict
        resSet = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            id_ = entry.find('{http://www.w3.org/2005/Atom}id').text
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            categories = [category.get('term') for category in entry.findall('{http://www.w3.org/2005/Atom}category')]
            publishDate = entry.find('{http://www.w3.org/2005/Atom}published').text
            links = [link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link')]

            article = Article(
                title=title,
                summary=summary,
                categories=categories,
                publishDate=publishDate,
                link=[l for l in links if 'pdf' in l][0]
            )
            # check if published date is in range
            if minDate and maxDate:
                if minDate <= datetime.datetime.strptime(publishDate, "%Y-%m-%dT%H:%M:%SZ") <= maxDate:
                    resSet.append(article)
            else:
                resSet.append(article)
        return resSet
    

    def get_full_article_text(self, link: str, max_tokens: int = 16000) -> str:
        '''get the text from the pdf link as raw text'''
        # get the pdf link
        response = requests.get(link)
        # convert the res to raw text
        fname = "temp.pdf"
        with open(fname, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(fname)
        content = loader.load_and_split()
        # delete the file
        os.remove(fname)
        outputStr = ""
        for doc in content:
            # cut of the aritcle when we hit the references section
            if "References" in doc.page_content:
                break
            # make sure summary doesn't exceed the model's context window
            elif (len(outputStr) / 4) > max_tokens * 0.80:
                break
            outputStr += doc.page_content
            outputStr += "\n\n"
        return outputStr



'''if __name__ == "__main__":
    api = arxivApi()
    maxDate = datetime.datetime.now()
    minDate = maxDate - datetime.timedelta(days=7)
    print(api.search_category("cs.AI", minDate=minDate, maxDate=maxDate))'''
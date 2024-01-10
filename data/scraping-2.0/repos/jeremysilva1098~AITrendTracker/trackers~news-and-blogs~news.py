import requests
import json
from dotenv import load_dotenv
import os
from data_models import Article
import pathlib
from typing import List, Optional
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from dotenv import load_dotenv

# load in the env variables
'''par_dir = pathlib.Path(__file__).parent.parent
dotenv_dir = f"{par_dir}/.env"
print("Reading .env variables from: ", dotenv_dir)
load_dotenv(dotenv_path=dotenv_dir)'''


class newsApi:
    def __init__(self) -> None:
        self.news_key = os.getenv("NEWS_API_KEY")
        self.news_url = "https://newsapi.org/v2/everything"
    

    def get_url_content(self, url: str) -> str:
        loader = UnstructuredURLLoader([url])
        #loader = SeleniumURLLoader([url])
        data = loader.load()
        content = ""
        for page in data:
            content += page.page_content
        return content
    

    def search_keywords(self, keywords: str, num_results: int = 10,
                        minDate: Optional[str] = None,
                        maxDate: Optional[str] = None) -> List[Article]:
        params = {
            "q": keywords,
            "from": minDate,
            "to": maxDate,
            "language": "en",
            "sortBy": "relavancy",
            "pageSize": num_results * 2,
        }
        headers = {
            # add the key as beaer token
            "Authorization": f"Bearer {self.news_key}"
        }
        response = requests.get(self.news_url, params=params, headers=headers)
        if response.status_code != 200:
            print("Status code: ", response.status_code)
            raise Exception(f"An error occurred: {response.content}")
        # get the JSON
        res = json.loads(response.content)
        # create output
        resSet = []
        count = 0  # Counter to keep track of the number of results
        for article in res['articles']:
            title = article['title']
            source = article['source']['name']
            description = article['description']
            url = article['url']
            content = self.get_url_content(url)
            # make sure there is content
            if len(content) > 500:
                resSet.append(Article(
                    title=title,
                    source=source,
                    description=description,
                    url=url,
                    content=content
                ))
                count += 1  # Increment the counter
                if count == num_results:
                    break  # Exit the loop when the desired number of results is reached
            else:
                continue
        return resSet
"""This module is responsible for loading the data from the raw data source. """
from pprint import pprint
from langchain.document_loaders import JSONLoader

class ArticleLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = JSONLoader(
            file_path=self.file_path,
            jq_schema='.[].text'
        )

    def load(self):
        data = self.loader.load()
        return data

if __name__ == "__main__":
    FILE_PATH = './data/raw/articles.json'
    article_loader = ArticleLoader(FILE_PATH)
    data = article_loader.load()
    pprint(data)

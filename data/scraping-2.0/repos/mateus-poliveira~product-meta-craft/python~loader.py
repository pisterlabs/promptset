from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
import asyncio

class Loader:
    def __init__(self, url):
        self.url = url
        self.extraction = []
    def extract(self): 
        # Load HTML
        loader = AsyncChromiumLoader([self.url])
        html = loader.load()
        print(html[0].page_content[0:100])
        print("-------------Got doc?------------------")
        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["h1","h2", "h3" ,"title"])
        # Result
        self.extraction = docs_transformed[0].page_content[0:2000]

def main():
    loader = Loader("https://www.ebay.com/b/VMAXTANKS-Rechargeable-Batteries/48619/bn_7114644579")
    loader.extract()
    print("------------Extraction-------------------")
    print(loader.extraction)
if __name__ == "__main__":
    main()

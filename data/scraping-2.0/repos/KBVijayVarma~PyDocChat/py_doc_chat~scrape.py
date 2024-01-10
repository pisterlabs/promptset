from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer

def scrape(doc_link):
    loader = AsyncHtmlLoader([doc_link])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["span","p"])
    return docs_transformed
from langchain.document_loaders import WebBaseLoader

def scrape_webpage(URL: str):
    """
    Scrape a webpage and return the document.
    
    Args:
        URL (str): URL of the webpage to be scraped.
    
    Returns:
        Document: Document object containing the scraped webpage."""
    loader = WebBaseLoader(URL)
    doc = loader.load()
    return doc

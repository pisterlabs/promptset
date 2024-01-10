# Importing necessary libraries
from langchain.loader import SitemapLoader, WebBaseLoader
from langchain.parser import HTMLParser

# Define a function to load documents from a sitemap
def load_sitemap(sitemap_url):
    # Create an instance of SitemapLoader
    loader = SitemapLoader()
    # Load the sitemap
    sitemap = loader.load(sitemap_url)
    # Initialize a parser
    parser = HTMLParser()
    # Initialize a list to store chunks
    chunks = []
    # Loop through each URL in the sitemap
    for url in sitemap:
        # Load the webpage
        page = loader.load(url)
        # Parse the webpage to extract chunks
        parsed_chunks = parser.parse(page, tag="p")
        # Add metadata to each chunk
        for chunk in parsed_chunks:
            chunk.metadata.update({
                "source": "sitemap",
                "url": url,
                "type": "documentation",
                "title": chunk.metadata.get("title", "")
            })
        # Add the chunks to the list
        chunks.extend(parsed_chunks)
    # Return the list of chunks
    return chunks

# Define a function to load documents from a Github repo
def load_repo(repo_url):
    # Create an instance of WebBaseLoader
    loader = WebBaseLoader()
    # Load the repo
    repo = loader.load(repo_url)
    # Initialize a parser
    parser = HTMLParser()
    # Initialize a list to store chunks
    chunks = []
    # Loop through each file in the repo
    for file in repo:
        # Load the file
        page = loader.load(file)
        # Parse the file to extract chunks
        parsed_chunks = parser.parse(page, tag="p")
        # Add metadata to each chunk
        for chunk in parsed_chunks:
            chunk.metadata.update({
                "source": "repo",
                "url": file,
                "type": "file",
                "title": chunk.metadata.get("title", "")
            })
        # Add the chunks to the list
        chunks.extend(parsed_chunks)
    # Return the list of chunks
    return chunks

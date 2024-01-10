import arxiv
from bs4 import BeautifulSoup
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.vectorstores import FAISS
import os
import psycopg2 as psycopg
import requests

def exec_static_sql_query(sql_query):
    """
    Runs a PostgreSQL query that has no parameters and returns the result.
    """
    conn_str = os.getenv('POSTGRES_CONN_STR', None)
    conn = psycopg.connect(conn_str)
    cur = conn.cursor()
    cur.execute(sql_query)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# getting a list of arxiv papers summaries and their metadata formatted as strings
def get_arxiv_search_results(query: str, num_results: int = 5):
    client = arxiv.Client()
    search = arxiv.Search(
        max_results=num_results,
        query=query,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results_formatted = []
    for paper in client.results(search):
        authors_str = ", ".join([author.name for author in paper.authors])
        links_str = ", ".join([link.href for link in paper.links])
        published_str = paper.published.strftime('%B %d, %Y')
        results_formatted.append({
            "authors": authors_str,
            "id": paper.get_short_id(),
            "links": links_str,
            "published": published_str,
            "summary": paper.summary,
            "title": paper.title,
        })
    return [f"""Title: {result["title"]}
            
        ID: {result["id"]}
        
        Authors: {result["authors"]}

        Published: {result["published"]}

        Summary: {result["summary"]}

        Links: {result["links"]}
        
    """ for result in results_formatted]

def get_pdf_document_chunks(pdf_file_path: str, chunk_size: int = 25000):
    """Split a PDF document into chunks of x tokens, with 50 tokens of overlap between each chunk (better for API calls)"""
    loader = PyPDFLoader(pdf_file_path)
    # this helper splits the document into pages
    pages = loader.load_and_split()
    # initializing a vectors store using the OpenAI embeddings class
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    # splitting the document into chunks of x tokens, with 50 tokens of overlap between each chunk (better for API calls)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=25000, chunk_overlap=50)
    # getting the content of each page as a list of strings
    pages_content = [page.page_content for page in pages]
    docs = text_splitter.create_documents(pages_content)
    return docs

def get_postgre_db_schema():
    return exec_static_sql_query("""SELECT
        table_name,
        column_name,
        column_default,
        data_type,
        is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'""")

def get_serp_links(query: str, num_results: int = 10):
    ddg_search = DuckDuckGoSearchAPIWrapper()
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

def scrape_webpage_text(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            # BeautifulSoup transforms a complex HTML document into a tree of Python objects,
            # such as tags, navigable strings, or comments
            soup = BeautifulSoup(r.text, 'html.parser')
            # separating all extracted text with a space
            text = soup.get_text(separator=" ", strip=True)
            return text
        else:
            return f"failed to scrape webpage with status: {r.status_code}"
    except Exception as e:
        return f"failed to scrape webpage with error:\n{e}"

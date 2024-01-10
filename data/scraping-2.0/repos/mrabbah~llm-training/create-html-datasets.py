import os
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_transformers import Html2TextTransformer

# function that get all html files in path_to_dir
def get_html_files(path_to_dir, list_folders_to_skip=[".git", ".github", "_images", "_static", "_attachments"]):
    html_files = []
    for root, dirs, files in os.walk(path_to_dir):
        # Exclude folders in list_folders_to_skip
        dirs[:] = [d for d in dirs if d not in list_folders_to_skip]
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files

# function that takes as argument a list of html files and returns a list of docs using UnstructuredHTMLLoader
def get_docs(html_files):
    docs = []
    for html_file in html_files:
        loader = UnstructuredHTMLLoader(html_file)
        doc = loader.load()[0]
        docs.append(doc)
    return docs

path_to_dir = "C:\\Users\\mrabb\\Documents\\GitHub\\corteza-html-docs"
html_files = get_html_files(path_to_dir, list_folders_to_skip=[".git", ".github", "_images", "_static", "_attachments"])
print(len(html_files))
# docs = get_docs(html_files)
# print(len(docs))



from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from bs4 import BeautifulSoup as Soup

domain_to_crawl = "https://docs.cortezaproject.org/corteza-docs/2023.9/"
loader = RecursiveUrlLoader(
    url=domain_to_crawl, 
    max_depth=2, 
    # extractor=lambda x: Soup(x, "html.parser").text,
    use_async=True,
    timeout=30,
    prevent_outside=True,
)
docs = loader.load()
print(len(docs))
print(docs[-1].page_content)
print(docs[-1].metadata)
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
print(len(docs_transformed))
print(docs_transformed[-1].page_content)
print(docs_transformed[-1].metadata)

# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin
# import mimetypes

# def is_html_content(response):
#     content_type = response.headers.get('Content-Type', '')
#     return 'text/html' in content_type

# def get_urls_recursive(domain, max_depth, current_depth=0, visited_urls=set()):
#     if current_depth > max_depth:
#         return []

#     try:
#         response = requests.get(domain)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         visited_urls.add(domain)

#         urls = set()
#         for a_tag in soup.find_all('a', href=True):
#             href = a_tag['href']
#             absolute_url = urljoin(domain, href)
#             parsed_url = urlparse(absolute_url)

#             if parsed_url.netloc == urlparse(domain).netloc and absolute_url not in visited_urls  and is_html_content(requests.get(absolute_url)) and not any(parsed_url.path.endswith(ext) for ext in mimetypes.types_map.values()):
#                 urls.add(absolute_url)
#                 urls.update(get_urls_recursive(absolute_url, max_depth, current_depth + 1, visited_urls))

#         return list(urls)

#     except Exception as e:
#         print(f"Error fetching {domain}: {e}")
#         return []
    
# domain_to_crawl = "https://docs.cortezaproject.org/corteza-docs/2023.9/index.html"
# max_depth_to_crawl = 2

# urls = get_urls_recursive(domain_to_crawl, max_depth_to_crawl)
# print(len(urls))

# from langchain.document_loaders import AsyncHtmlLoader

# # function that retrieves all sub urls for a given domain recursively and returns docs
# def get_docs(domain, max_depth=2, current_depth=0, urls=[]):
#     if current_depth == max_depth:
#         return urls
#     else:
#         current_depth += 1
#         loader = AsyncHtmlLoader([domain])
#         docs = loader.load()
#         for doc in docs:
#             print(doc)
#             urls += doc.urls
#         return get_urls(domain, max_depth, current_depth, urls)

# domain = "https://docs.cortezaproject.org/corteza-docs/2023.9/"
# urls = get_urls(domain, max_depth=2)
# print(len(urls))
# urls = []
# loader = AsyncHtmlLoader(urls)
# docs = loader.load()
# print(len(docs))


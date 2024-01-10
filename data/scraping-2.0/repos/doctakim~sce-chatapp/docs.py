"""
docs.py scrapes the website and stores the documents in a list.
"""

import xmltodict
import requests
# import nest_asyncio
from langchain.document_loaders.sitemap import SitemapLoader

# nest_asyncio.apply()

url = "https://www.sce.com"
url_site_map = f"{url}/sitemap.xml"
filter_urls = [
    "https://www.sce.com/residential/rebates-savings/summer-discount-plan/"
]

sitemap_loader = SitemapLoader(
    web_path=url_site_map,      
    filter_urls=filter_urls,
  )

docs = sitemap_loader.load()
# sitemap_loader.requests_per_second = 2
# # Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
# sitemap_loader.requests_kwargs = {"verify": False}
# дозагрузка отсутствующих файлов

from parsing.SiteLoader import list_docs_toload, saveDocs_toload, dpages, SiteLoader
import nest_asyncio
nest_asyncio.apply()
from langchain.document_loaders.sitemap import SitemapLoader
import os

def reloadhtml():
    docsload = list_docs_toload()
    pages = list(map(lambda x: x.page, docsload))
    url_site_map=""
    sitemap_loader = SitemapLoader(url_site_map)
    docs = sitemap_loader.scrape_all(pages)
    saveDocs_toload(docs, docsload)

def htmltomd(i):
    num = dpages[i]  # str(i).rjust(2,"0")
    filename = num.md_file()
    print(filename)
    sl = SiteLoader(i)
    sl.parse()
    sl.save()



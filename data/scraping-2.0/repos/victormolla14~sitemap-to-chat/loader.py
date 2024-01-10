# loader.py

from langchain.document_loaders.sitemap import SitemapLoader
import nest_asyncio

class DocumentLoader:
    def __init__(self, sitemap):
        self.sitemap = sitemap
        nest_asyncio.apply()
        self.sitemap_loader = SitemapLoader(web_path=sitemap)

    def load_document(self):
        try:
            document = self.sitemap_loader.load()
            print(len(document[0].page_content))
            return document
        except Exception as e:
            print(f"Se produjo un error: {e}")
            return None

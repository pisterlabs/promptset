from langchain.document_loaders.sitemap import SitemapLoader
from config import setup_logging

# Set up logging
logger = setup_logging()

def load_sitemap(sitemap_path):
    logger.info(f'Loading sitemap for {sitemap_path}')
    sitemap_loader = SitemapLoader(web_path=sitemap_path)
    sitemap_data = sitemap_loader.load()
    return sitemap_data

from langchain.document_loaders import PlaywrightURLLoader

from utils import logger


async def website_text_data(urls):
    loader = PlaywrightURLLoader(urls)
    loader.requests_kwargs = {'verify': False}

    logger.info(f"Scraping websites: {urls}")

    data = await loader.aload()

    logger.info("Websites scraped successfully")

    return data

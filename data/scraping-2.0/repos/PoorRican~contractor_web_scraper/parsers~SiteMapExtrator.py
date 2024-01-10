from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from llm import LONG_MODEL_PARSER, LONG_LLM
from typedefs.sitemap import SiteMap
from utils import fetch_site, strip_url

_parser = PydanticOutputParser(pydantic_object=SiteMap)


_extract_prompt = PromptTemplate.from_template(
    """You will be given several <a> tags from a company website, {url}
    
    Here are the <a> tags:
    ```{links}```
    
    Extract the 'About Us' URL and the 'Contact Us' URL.
    
    {format_instructions}
    """,
    partial_variables={'format_instructions': _parser.get_format_instructions()}
)


class SiteMapExtractor:

    _chain = _extract_prompt | LONG_LLM | _parser

    @staticmethod
    def _preprocess_links(links: list[str], url: str) -> list[str]:
        """ Preprocess the links before passing to the LLM model.

        Any absolute URL that is external to the site is removed.
        Any relative links are converted to absolute links.
        """
        base = strip_url(url)
        processed = []
        for link in links:
            if link.startswith('http'):
                if base in link:
                    processed.append(link)
            else:
                processed.append(base + link)
        return processed

    @classmethod
    async def extract(cls, url: str) -> SiteMap:
        """ Extract all links from the page """
        content = await fetch_site(url)

        links = []
        for anchor in content.find_all('a'):
            if 'href' in anchor.attrs:
                links.append(anchor.attrs['href'])

        links = cls._preprocess_links(links, url)
        extracted = await cls._chain.ainvoke({'links': links, 'url': url})
        return extracted

    @classmethod
    async def __call__(cls, url: str) -> SiteMap:
        return await cls.extract(url)

import asyncio
from asyncio import sleep
from copy import copy
from typing import Union, Generator, ClassVar, Generic, TypeVar

from bs4 import Tag
from langchain.schema.output_parser import OutputParserException
from langchain.schema.runnable import Runnable
from openai import InvalidRequestError
from openai.error import RateLimitError

from log import logger
from typedefs import ContractorCallback, LLMInput
from utils import strip_html_attrs


_SLEEP_TIME = 1
""" Time to sleep in seconds after a rate limit error. """

T = TypeVar('T')


class TextSnippetScraper(Generic[T]):
    """ A template functor class for scraping text snippets from HTML content.

    This class is designed for extracting small text snippets from HTML content. It begins by looking at the header and
    footer of the HTML content, then looks at all small text snippets (<p>, <span>, <a>). Finally, it then chunks takes
    large chunks of text from the beginning and end of the HTML content.

    Subclasses should override the following properties:
        _chain: a chain of LLM models to extract data from text snippets
        _failure_text: a string to be returned by LLM when no data is found
        _search_type: a string representing the type of data being scraped. Used for user feedback.
    """

    _chunk_size: ClassVar[int] = 100
    """ Number of `Tag` objects in a single batch to process at once. """

    _chain: ClassVar[Runnable]
    _failure_text: ClassVar[str]
    _search_type: ClassVar[str]

    @classmethod
    async def _process(cls, content: LLMInput) -> Union[T, None]:
        """ Attempt to extract the text snippet from HTML content using `self._chain`.

        There is an internal mechanism that will retry the extraction if a rate limit error is encountered.

        Parameters:
            content: HTML content to extract snippet from

        Returns:
            The extracted snippet. `None` if no snippet was found or an error was encountered.
        """
        max_retries = 15
        retries = 0
        while retries < max_retries:
            try:
                result: str = await cls._chain.ainvoke({'content': str(content)})
                is_str = type(result) is str
                if not is_str or (is_str and cls._failure_text not in result.lower()):
                    return result
                break

            except OutputParserException:
                break

            except RateLimitError:
                retries += 1

                if retries == max_retries:
                    logger.error("Rate limit hit too many times. Aborting...")
                else:
                    logger.warning(f"Rate limit hit. Retrying {cls._search_type} extraction after {_SLEEP_TIME}s...")
                    await sleep(_SLEEP_TIME)

            except InvalidRequestError:
                # TODO: break down content into smaller chunks
                logger.error(f"InvalidRequestError while scraping {cls._search_type}. "
                             "String might be too many tokens...")
                break

        return None

    @classmethod
    def _snippet_chunks(cls, content: Tag) -> Generator[list[Tag], None, None]:
        """ Iterate over all small text snippets in the HTML content and yield them in chunks.

        Parameters:
            content: HTML content to iterate over

        Yields:
            A list of small text snippets in chunks according to `TextSnippetScraper._chunk_size`
        """
        chunks = []
        for i in ('p', 'span', 'a', 'strong', 'li', 'b', 'u', 'font'):
            sections = content.find_all(i)
            for section in sections:
                chunks.append(section)
                if len(chunks) >= cls._chunk_size:
                    yield chunks
                    chunks = []

    @classmethod
    async def _process_chunks(cls, chunks: list[Tag], callback: ContractorCallback) -> bool:
        """ Simultaneously process a list of chunks and call the callback function if a snippet is found.

        Parameters:
            chunks: list of chunks to process
            callback: callback function to execute if snippet is found

        Returns:
            True if snippet was found, False otherwise
        """
        coroutines = await asyncio.gather(*[cls._process(chunk) for chunk in chunks])
        for result in coroutines:
            if result is not None:
                callback(result)
                return True
        return False

    async def __call__(self, content: Tag, url: str, callback: ContractorCallback) -> bool:
        """ Scrape snippet from HTML content.

        This will attempt to scrape a snippet from the HTML content. If a snippet is found, it will be passed to the
        callback function. If no snippet is found, a warning will be raised.

        Parameters:
            content: HTML content to scrape snippet from
            url: URL of the HTML content. This is used in the warning message.
            callback: callback function to pass snippet to

        Returns:
            True if snippet was found, False otherwise
        """
        _content = strip_html_attrs(copy(content))

        # attempt to find snippet in footer or header
        for i in ('footer', 'header'):
            section = _content.find(i)
            if section is not None:
                snippet = await self._process(section)
                if snippet is not None:
                    callback(snippet)
                    return True

        # begin to look at all small text snippets
        for chunk in self._snippet_chunks(_content):
            if await self._process_chunks(chunk, callback):
                return True

        # as a last resort, look at first and last chunks
        chunk_size = 5000
        first, last = str(_content)[:chunk_size], str(_content)[-chunk_size:]
        for i in (first, last):
            snippet = await self._process(i)
            if snippet is not None:
                callback(snippet)
                return True

        logger.debug(f"Could not extract {self._search_type} snippet from '{url}'")

        return False

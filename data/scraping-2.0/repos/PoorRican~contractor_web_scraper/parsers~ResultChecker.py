from langchain.prompts import PromptTemplate

from log import logger
from llm import MODEL_PARSER
from typedefs import SearchResult
from utils import strip_url

_description_expand_prompt = PromptTemplate.from_template(
    """ You will be given the title, URL, and description of a search result.

    Here is the title: {title}
    Here is the URL: {url}
    Here is the description: {description}


    Please explain what this page is about in one sentence.
    Does this page directly represent a contractor company website?
    """
)

_is_contractor_prompt = PromptTemplate.from_template(
    """Given an explanation of a search result,
    determine if it directly represents a webpage for a construction contractor company website.
    Return 'contractor' if it is,
    and 'not contractor' if it does not directly represent a contractor company website.

    {explanation}
    """
)


_expand_chain = _description_expand_prompt | MODEL_PARSER


class ResultChecker:
    """ Functor which checks if a `SearchResult` is a valid company by comparing the `title` and `url`.

    If the `url` seems to be valid for the given company, then `True` is given. Otherwise, `False` is returned.
    This is meant to be used by `SearchHandler` to ignore sites which feature valid companies such as local newspapers
    and instead favor actual company websites.
    """

    _chain = {'explanation': _expand_chain} | _is_contractor_prompt | MODEL_PARSER

    @staticmethod
    def _is_contractor(response: str) -> bool:
        """ Detect if site is a company site based on search result.

        This is parse the result from `_is_contractor_chain`.

        Parameters:
            response: LLM response from `_is_contractor_chain`

        Returns:
            True if site is a contractor site, False otherwise
        """
        if 'not contractor' in response.lower():
            return False
        elif 'contractor' in response.lower():
            return True
        else:
            raise ValueError(f"`_is_contractor_chain` returned ambiguous output: '{response}'")

    @classmethod
    async def __call__(cls, result: SearchResult) -> bool:
        title = result.title
        description = result.description
        url = strip_url(result.url)

        response = await cls._chain.ainvoke({
            'title': title,
            'description': description,
            'url': url,
        })
        is_contractor = cls._is_contractor(response)

        if not is_contractor:
            logger.debug(f"Rejected '{title}': {url} - Description does not indicate contractor site!")
        return is_contractor

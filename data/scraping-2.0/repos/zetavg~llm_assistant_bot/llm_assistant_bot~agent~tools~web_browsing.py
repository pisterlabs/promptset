from typing import Type, Optional, Union, Tuple, List, Dict

import logging
from urllib.parse import quote
from pydantic import BaseModel, Field

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    aget_current_page,
    )
from langchain.agents.agent_toolkits.playwright.toolkit import (
    BaseBrowserTool,
    ClickTool,
    CurrentWebPageTool as OriginalCurrentWebPageTool,
    ExtractHyperlinksTool as OriginalExtractHyperlinksTool,
    ExtractTextTool as OriginalExtractTextTool,
    GetElementsTool,
    NavigateTool as OriginalNavigateTool,
    NavigateBackTool as OriginalNavigateBackTool,
)

from readability import Document
from markdownify import markdownify
from bs4 import BeautifulSoup


from ...config import Config

logger = logging.getLogger("web_browsing_tool")


class NavigateTool(OriginalNavigateTool):
    name: str = "browser_navigate"
    description: str = "Navigate the browser to the specified URL."

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        logger.debug(f"Navigating to '{url}' ...")

        try:
            output = await super()._arun(url=url, run_manager=run_manager)
            page = await aget_current_page(self.async_browser)  # type: ignore
            html_content = await page.content()

            doc = Document(html_content)
            output += f'\n----\nContent:\n{doc.title()}\n{markdownify(doc.summary())}'

            # output = output[:1024]

            output_for_logging = output.replace('\n', '\\n')[:200]
            logger.debug(f"Results of '{url}': {output_for_logging}")

            output += '\n----\n'
            # output += "Hint: you can use the browser_extract_current_page_text tool to get the full content of this page"

            return output
        except Exception as e:
            return str(e)


class GoogleSearchToolInput(BaseModel):
    keyword: str = Field(..., description="keyword(s) for searching")


class GoogleSearchTool(OriginalNavigateTool):
    name: str = "browser_google_search"
    description: str = "Search the specified keywords on Google. Do not use this if unnecessary, prefer other tools first and think if you can do it without Google. Also, do not use this tool to do translations. This tool should only be used as a last resort if you can't find any available data from your memory or other tools."
    args_schema: Type[BaseModel] = GoogleSearchToolInput

    async def _arun(
        self,
        keyword: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        logger.debug(f"Searching Google for '{keyword}' ...")

        url = f"https://www.google.com/search?q={quote(keyword)}"
        output = await super()._arun(url=url, run_manager=run_manager)

        page = await aget_current_page(self.async_browser)  # type: ignore
        html_content = await page.content()

        # doc = Document(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        search_results_element = soup.find(id='search')

        # import pdb
        # pdb.set_trace()

        if not search_results_element:
            output += '\n'
            output += 'Cannot parse search result'
            return output

        link_elements = search_results_element.find_all(
            lambda tag: (
                tag.has_attr('href')
                and tag['href'].startswith('http')
                and not tag['href'].startswith(
                    'https://www.google.com/search'
                )
                and not tag['href'].startswith(
                    'https://translate.google.com'
                )
            )
        )

        def find_description(elem, max_depth=4):
            current_depth = 0
            parent_elem = elem

            while current_depth <= max_depth:
                parent_elem = parent_elem.parent
                if not parent_elem:
                    break
                if parent_elem.has_attr('lang'):
                    break

            if not parent_elem:
                return None

            container_elem = parent_elem.find(
                lambda tag: (
                    tag.has_attr('data-sncf')
                    and tag['data-sncf'].strip().startswith('1')
                    )
                )
            if not container_elem:
                return None

            return container_elem.text.strip()

        results = [
            (elem.text, elem['href'], find_description(elem))
            for elem in link_elements
        ]

        # import pdb
        # pdb.set_trace()

        logger.debug(f"Google search results of '{keyword}': {results}.")

        search_results = []

        for title, url, description in results:
            if len(''.join(search_results)) > 2000:
                break

            info = [
                f"Title: {title[:100]}",
                f"URL: {url}",
            ]
            if description:
                info.append(f"Description: {description[:200]}")
            search_results.append('\n'.join(info))

        output = '\n----\n'.join([output] + search_results[:8])

        output += '\n----\n'
        output += "Hint: you can use the browser_navigate tool to navigate to the URLs."

        return output


# class CurrentWebPageTool(OriginalCurrentWebPageTool):
#     name: str = "browser_current_webpage"

#     # To avoid StopIteration error raised at
#     # https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L185
#     args_schema: Type[BaseModel] = None  # type: ignore

#     # Force this to return empty tuple and empty dict since the language model
#     # might pass '' instead of None as the input, which can cause an error
#     # ("TypeError: _arun() got multiple values for argument 'run_manager'")
#     # on https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L330.
#     def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
#         return (), {}


# class ExtractHyperlinksTool(OriginalExtractHyperlinksTool):
#     # To avoid StopIteration error raised at
#     # https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L185
#     args_schema: Type[BaseModel] = None  # type: ignore

#     # Always use absolute URL.
#     def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
#         return (True,), {}

#     # Need to limit the length of the output since the language model might
#     # blow up.
#     async def _arun(
#         self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
#     ) -> str:
#         output = await super()._arun(run_manager=run_manager)
#         return output[:2048]


class GetPageContentTool(OriginalExtractTextTool):
    name: str = "browser_extract_current_page_text"
    description: str = "Extract the full text content on the current webpage, before using this tool, you should use browser_navigate to navigate to the desired page."
    # To avoid StopIteration error raised at
    # https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L185
    args_schema: Type[BaseModel] = None  # type: ignore

    # Force this to return empty tuple and empty dict since the language model
    # might pass '' instead of None as the input, which can cause an error
    # ("TypeError: _arun() got multiple values for argument 'run_manager'")
    # on https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L330.
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    # Use readability to get the main content of the page.
    async def _arun(
        self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        page = await aget_current_page(self.async_browser)  # type: ignore
        html_content = await page.content()

        doc = Document(html_content)

        output = f'Title: "{doc.title()}", Content:\n{markdownify(doc.summary())}'

        return output[:4096]


# class NavigateBackTool(OriginalNavigateBackTool):
#     # To avoid StopIteration error raised at
#     # https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L185
#     args_schema: Type[BaseModel] = None  # type: ignore

#     # Force this to return empty tuple and empty dict since the language model
#     # might pass '' instead of None as the input, which can cause an error
#     # ("TypeError: _arun() got multiple values for argument 'run_manager'")
#     # on https://github.com/hwchase17/langchain/blob/cf5803e/langchain/tools/base.py#L330.
#     def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
#         return (), {}


browser_tools_classes: List[Type[BaseBrowserTool]] = [
    # ClickTool,
    GoogleSearchTool,
    NavigateTool,
    # NavigateBackTool,
    # GetPageContentTool,
    # ExtractHyperlinksTool,
    # GetElementsTool,
    # CurrentWebPageTool,
]


def get_async_browser():
    async_browser = create_async_playwright_browser()
    return async_browser


def get_browser_tools(async_browser=None):
    if not async_browser:
        async_browser = get_async_browser()

    browser_tools = [
        tool_cls.from_browser(
            sync_browser=None, async_browser=async_browser,
        )
        for tool_cls in browser_tools_classes
    ]

    return browser_tools

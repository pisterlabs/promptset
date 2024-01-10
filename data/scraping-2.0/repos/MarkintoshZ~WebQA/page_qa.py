import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import _load_map_reduce_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from html2text2 import html2text

load_dotenv()
logger = logging.getLogger(__name__)


class URLLoader(BaseLoader):
    """Loader that uses Playwright to load the page html and use readability.js
    and html2text library to sanitize and convert the raw html to markdown.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        headless (bool): If True, the browser will run in headless mode.

    Note: this loader is converted from the original PlaywrightURLLoader
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        headless: bool = True,
        remove_selectors: Optional[List[str]] = None,
    ):
        """Load a list of URLs using Playwright and unstructured."""
        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headless = headless
        self.remove_selectors = remove_selectors

    def load(self) -> List[Document]:
        """Load the specified URLs using Playwright and create Document instances.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """

        docs: List[Document] = list()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            for url in self.urls:
                try:
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")

                    for selector in self.remove_selectors or []:
                        elements = page.locator(selector).all()
                        for element in elements:
                            if element.is_visible():
                                element.evaluate("element => element.remove()")

                    page_source = page.content()
                    # readabilipy is used to remove scripts and styles
                    # simple_tree = simple_tree_from_html_string(page_source)

                    soup = BeautifulSoup(
                        "".join(s.strip() for s in page_source.split("\n")),
                        "html.parser",
                    )

                    for s in soup.select("script"):
                        s.extract()
                    for s in soup.select("style"):
                        s.extract()

                    def add_divider(node, threshold):
                        if isinstance(node, str):
                            return
                        tags = set()
                        children_count = set()
                        for child in node.children:
                            if child == "\n":
                                child.decompose()

                            add_divider(  # pylint: disable=cell-var-from-loop
                                child, threshold
                            )
                            tags.add(child.name)
                            if hasattr(child, "contents"):
                                children_count.add(len(child.contents))

                        if (
                            node.name
                            not in {
                                "ul",
                                "ol",
                                "table",
                                "tbody",
                                "thead",
                                "tr",
                                "td",
                                "th",
                            }
                            and len(tags) == 1
                            and len(children_count) == 1
                            and len(node.contents) >= threshold
                        ):
                            for i in range(-len(node.contents) + 1, 0, 1):
                                new_tag = soup.new_tag(
                                    "p"
                                )  # pylint: disable=cell-var-from-loop
                                new_tag.string = "---"
                                node.insert(i, new_tag)

                    add_divider(soup.body, 3)
                    simple_tree = soup.prettify()
                    print(simple_tree)
                    # html2text is used to convert html to markdown
                    text = html2text(str(simple_tree))
                    metadata = {"source": url}
                    docs.append(Document(page_content=text, metadata=metadata))
                except Exception as err:  # pylint: disable=broad-except
                    if self.continue_on_failure:
                        logger.error(
                            "Error fetching or processing %s, exception: %s", url, err
                        )
                    else:
                        raise err
            browser.close()
        return docs


system_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim. If there is no relevant text, say exactly "None." and nothing more.
______________________
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)


system_template = """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)


def answer(
    query: str,
    url: Optional[str] = None,
    filename: Optional[str] = None,
    output_llm_input: bool = False,
):
    """Answer a question from reading a website"""

    assert (url is None) ^ (
        filename is None
    ), "Exactly one of url or html_path must be provided"

    if url:
        loader = URLLoader(urls=[url], headless=True)
        page_content = loader.load()[0].page_content
    elif filename:
        with open(filename, "r", encoding="utf-8") as f:
            page_content = f.read()
    else:
        # unreachable code but use else branch to fix linter complaint
        page_content = ""

    if output_llm_input:
        print(page_content)

    llm_map = ChatOpenAI(temperature=0)  # type: ignore
    llm_reduce = ChatOpenAI()  # type: ignore

    qa_chain = _load_map_reduce_chain(
        llm_map,
        question_prompt=CHAT_QUESTION_PROMPT,
        combine_prompt=CHAT_COMBINE_PROMPT,
        reduce_llm=llm_reduce,
        verbose=True,
    )
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    output = qa_document_chain(
        {"input_document": page_content, "question": query},
        return_only_outputs=True,
    )
    print("Answer:", output["output_text"])


def download(url: str, filename: str, output_content: bool = False):
    """Download a webpage to a file"""
    loader = URLLoader(
        urls=[url],
        headless=True,
        remove_selectors=["script", "style"],
        continue_on_failure=False,
    )
    page_content = loader.load()[0].page_content
    with open(filename, "w", encoding="utf-8") as f:
        f.write(page_content)
    if output_content:
        print(page_content)


if __name__ == "__main__":
    import fire

    fire.Fire({"ans": answer, "dl": download})

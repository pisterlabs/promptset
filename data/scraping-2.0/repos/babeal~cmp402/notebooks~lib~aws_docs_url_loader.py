import logging
from typing import Dict, List, Union
import re

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class AWSDocsURLLoader(BaseLoader):
    """Load `HTML` files and parse them with `beautiful soup`."""

    def __init__(
        self,
        urls: List[str],
        mode: str = "md",
        show_progress_bar: bool = False,
        continue_on_failure: bool = True,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            urls: An array of the urls to load
            mode: text or markdown (md | text)
            show_progress_bar: whether to show the progress bar
            get_text_separator: The separator to use when calling get_text on the soup.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with " "`pip install beautifulsoup4`"
            )
        try:
            import markdownify
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with " "`pip install beautifulsoup4`"
            )
        try:
            import requests
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with " "`pip install beautifulsoup4`"
            )

        self.urls = urls
        self.mode = mode
        self.continue_on_failure = continue_on_failure
        self.show_progress_bar = show_progress_bar
        self.get_text_separator = get_text_separator

    def __replace_newline_and_spaces(self, text):
        # Replace occurrences of '\n' followed by any number of spaces with a single space
        return re.sub(r"\n\s*", " ", text)

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup
        from markdownify import MarkdownConverter
        import requests

        docs: List[Document] = list()
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e

            urls = tqdm(self.urls)
        else:
            urls = self.urls

        for url in urls:
            try:
                response = requests.get(
                    url,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0"}, # Set user agent to mobile to get mobile version of the page
                )  
                response.raise_for_status()  # Raise an error if the response is not successful
                response.encoding = "UTF-8"
                response_fixed = self.__replace_newline_and_spaces(response.text)
                soup = BeautifulSoup(response_fixed, "html.parser")

                main_content = soup.find("div", {"id": "main-content"})
                if self.mode == "md":
                    text = MarkdownConverter(heading_style="ATX").convert_soup(main_content)
                elif self.mode == "text":
                    text = main_content.get_text()
                else:
                    raise ValueError("Mode can only be md or text")

            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                    continue
                else:
                    raise e

            if soup.title:
                title = str(soup.title.string)
            else:
                title = ""

            metadata: Dict[str, Union[str, None]] = {
                "source": url,
                "title": title,
            }
            docs.append(Document(page_content=str(text), metadata=metadata))

        return docs

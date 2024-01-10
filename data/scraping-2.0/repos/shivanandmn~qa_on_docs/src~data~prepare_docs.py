from langchain.document_loaders.pdf import BasePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
    Dict,
    List,
)

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document
import pdfplumber
from tqdm import tqdm


class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
        self, text_kwargs: Optional[Mapping[str, Any]] = None, dedupe: bool = False
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document
            for page in tqdm(doc.pages):
                if page.page_number >= self.text_kwargs.get(
                    "start_page", 0
                ) and page.page_number <= self.text_kwargs.get(
                    "end_page", len(doc.pages)
                ):
                    yield Document(
                        page_content=extract_cropped_text_from_page(
                            page=page,
                            left_margin=52,
                            right_margin=52,
                            top_margin=72,
                            bottom_margin=100,
                        ),
                        metadata=dict(
                            {
                                "source": blob.source,
                                "file_path": blob.source,
                                "page": page.page_number,
                                "total_pages": len(doc.pages),
                            },
                            **{
                                k: doc.metadata[k]
                                for k in doc.metadata
                                if type(doc.metadata[k]) in [str, int]
                            },
                        ),
                    )
                    # By default, pdfplumber keeps in cache to avoid to reprocess the same page, leading to memory issues.
                    page.flush_cache()

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)


class PDFPlumberLoader(BasePDFLoader):
    """Load `PDF` files using `pdfplumber`."""

    def __init__(
        self,
        file_path: str,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        headers: Optional[Dict] = None,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pdfplumber  # noqa:F401
        except ImportError:
            raise ImportError(
                "pdfplumber package not found, please install it with "
                "`pip install pdfplumber`"
            )

        super().__init__(file_path, headers=headers)
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe

    def load(self) -> List[Document]:
        """Load file."""

        parser = PDFPlumberParser(text_kwargs=self.text_kwargs, dedupe=self.dedupe)
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


def extract_cropped_text_from_page(
    page, left_margin: int, top_margin: int, right_margin: int, bottom_margin: int
) -> str:
    crop_box = (
        left_margin,
        top_margin,
        page.width - right_margin,
        page.height - bottom_margin,
    )
    text = page.crop(bbox=crop_box).extract_text(x_tolerance=1, y_tolerance=3)
    return text


def pdf_load_split(config):
    loader = PDFPlumberLoader(
        config.get("file"),
        text_kwargs={
            "start_page": 1,
            "end_page": 9,
            "header_height": 1,
            "footer_height": 1,
        },
    )
    pages = loader.load()
    chunk_size = config.get("chunk_size")
    chunk_overlap = config.get("chunk_overlap")
    separator = config.get("separator", "\n")
    text_splitter = CharacterTextSplitter(
        separator="\n" if separator is None else separator,
        chunk_size=300 if chunk_size is None else chunk_size,
        chunk_overlap=100 if chunk_overlap is None else chunk_overlap,
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)
    return docs


if __name__ == "__main__":
    pdf_load_split({"file": "data/2308.13418.pdf"})

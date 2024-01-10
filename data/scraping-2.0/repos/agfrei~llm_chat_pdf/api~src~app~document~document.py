import re

from bs4 import BeautifulSoup
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma

from src.app.core.settings import Settings


class Document:
    def __init__(self, path: str, settings: Settings):
        self._path = path
        self._pages = None
        self._settings = settings
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._settings.chunk_markdown_separators,
            return_each_line=False,
        )
        self._rct_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.chunk_separators,
        )
        self.chunks = []

    def load(self):
        """Load PDF into meanigful chunks.

        Strategy:
        1- Load PDF as HTML using PyMuPDF
        2- Split each `div` and `span` into `sections` with
           font-size as metadata
        3- Convert into markdow using font size to infer headers
           (bigger fonts = top headers, lowest font = simple text)
        4- Use `MarkdownHeaderTextSplitter` to split markdown into
           meanigful chunks
        5- Use `RecursiveCharacterTextSplitter` to split
        """
        loader = PyMuPDFLoader(file_path=self._path)
        self._pages = loader.load(option="html")

        html_sections = []
        font_sizes = set()
        for page in self._pages:
            s, fs = self.__split_html_sections(page)
            html_sections.extend(s)
            font_sizes = font_sizes.union(fs)

        markdown = self.__get_markdown(html_sections, font_sizes)
        chunks = self._md_splitter.split_text(markdown)

        self.chunks = []
        for i, chunk in enumerate(chunks):
            smaller_chunks = self._rct_splitter.split_documents([chunk])

            for j, c in enumerate(smaller_chunks):
                header_append = (
                    "| "
                    + " ".join(
                        [
                            c.metadata.get(header, "")
                            for _, header in self._settings.chunk_markdown_separators  # noqa: E501
                        ]
                    ).strip()
                    + " |"
                )
                if header_append:
                    c.page_content = header_append + " " + c.page_content

                c.metadata["md_section"] = i + 1
                c.metadata["total_md_sections"] = len(chunks)
                c.metadata["chunk_split"] = j + 1
                c.metadata["total_chunk_splits"] = len(smaller_chunks)

                self.chunks.append(c)

        self.__add_to_vector_db()

    def __split_html_sections(self, page):
        soup = BeautifulSoup(page.page_content, "html.parser")
        content = soup.find_all("div")

        current_font_size = None
        current_text = ""
        snippets = []
        font_sizes = set()

        for c in content:
            span = c.find("span")
            if not span:
                continue

            while span:
                style = span.get("style")
                if not style:
                    span = span.findNext()
                    continue

                font_size = re.findall(
                    r"font-size:(\d+|\d+\.\d+)(pt|px)", style
                )
                if not font_size:
                    span = span.findNext()
                    continue

                font_size = int(float(font_size[0][0]))
                font_sizes.add(font_size)
                if not current_font_size:
                    current_font_size = font_size

                if font_size == current_font_size:
                    current_text += span.text + "\n"
                else:
                    snippets.append((current_text, current_font_size))
                    current_font_size = font_size
                    current_text = span.text + "\n"

                span = span.findNext()

        snippets.append((current_text, current_font_size))
        return snippets, font_sizes

    def __get_markdown(self, snippets: list, font_sizes: set):
        font_sizes = sorted(list(font_sizes), reverse=True)
        formatter = {}

        for i, size in enumerate(font_sizes):
            if i == len(font_sizes) - 1:
                format = ""
            else:
                format = (i + 1) * "#" + " "
            formatter[size] = format

        formatter

        snippets = [(formatter[s[1]] + s[0], s[1]) for s in snippets]

        markdown = ""
        for s in snippets:
            markdown += s[0]

        return markdown

    def __add_to_vector_db(self):
        embedding = OpenAIEmbeddings(
            openai_api_key=self._settings.openai_api_key
        )
        _ = Chroma.from_documents(
            self.chunks,
            embedding,
            persist_directory=self._settings.chroma_persist_directory,
        )

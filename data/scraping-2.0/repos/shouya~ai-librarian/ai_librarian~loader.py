import ebooklib
import ebooklib.epub
import epub_meta
from bs4 import BeautifulSoup as BS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base import Document, Loader

import pprint


class EpubBookLoader(Loader):
    epub = None
    chapters = None
    doc_index = {}

    def __init__(self, file_path):
        """Initialize a book loader"""
        self.file_path = file_path

    def load(self):
        """Parse the book file"""
        load_opts = {"ignore_ncx": True}
        self.epub = ebooklib.epub.read_epub(self.file_path, load_opts)
        self.chapters = self._parse_chapters(self.epub)

    def book_id(self):
        """Calculate the id of the book (sha1 hash of the file)"""
        import hashlib

        sha1 = hashlib.sha1()
        with open(self.file_path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()[:32]

    def to_docs(self):
        docs = []
        docs.extend(self.split_paragraph_docs())
        docs.extend(self.split_sentence_docs())
        return docs

    def _parse_chapters(self, epub):
        """Parse the chapters of the book"""
        chapters = []
        for item in epub.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            dom = BS(item.get_content(), "xml")
            title = dom.find("h1") or dom.find("h2") or dom.find("h3")
            if not title:
                continue
            title = title.text

            paragraphs = [p.text for p in dom.find_all("p")]
            if len(paragraphs) == 0 or len(paragraphs[0]) == 0:
                continue

            chapter = {"title": title, "paragraphs": paragraphs}
            chapters.append(chapter)

        for i in range(len(chapters)):
            chapters[i]["index"] = i
        return chapters

    def whole_chapter_docs(self):
        """Convert the book to a list of documents"""
        docs = []
        for chapter in self.chapters:
            doc = Document(
                id=f"whole_chapter:{chapter['index']}",
                content="\n\n".join(chapter["paragraphs"]),
                metadata={
                    "chapter_index": chapter["index"],
                    "chapter_title": chapter["title"],
                },
                embedding=None,
            )
            docs.append(doc)
        return docs

    def _split_docs(self, level, splitter_conf):
        splitter = RecursiveCharacterTextSplitter(**splitter_conf)
        docs = []

        for whole_doc in self.whole_chapter_docs():
            chapter_index = whole_doc.metadata["chapter_index"]
            split_docs = splitter.create_documents([whole_doc.content])

            for part_no, split_doc in enumerate(split_docs):
                metadata = whole_doc.metadata.copy()
                metadata["prev_id"] = None
                metadata["next_id"] = None

                part = f"{part_no+1}/{len(split_docs)}"
                id = f"{level}:{chapter_index}:{part}"

                content = split_doc.page_content

                doc = Document(
                    id=id,
                    content=content,
                    metadata=metadata,
                    embedding=None,
                )
                docs.append(doc)

        for i in range(1, len(docs)):
            docs[i].metadata["prev_id"] = docs[i - 1].id

        for i in range(0, len(docs) - 1):
            docs[i].metadata["next_id"] = docs[i + 1].id

        return docs

    def split_chapter_docs(self):
        return self._split_docs(
            "chapter", {"chunk_size": 2000, "chunk_overlap": 100}
        )

    def split_paragraph_docs(self):
        return self._split_docs(
            "paragraph", {"chunk_size": 800, "chunk_overlap": 0}
        )

    def split_sentence_docs(self):
        return self._split_docs(
            "sentence", {"chunk_size": 200, "chunk_overlap": 0}
        )


if __name__ == "__main__":
    loader = EpubBookLoader("/home/shou/tmp/book/book.epub")
    loader.parse_book()
    pprint.pprint(loader.split_chapter_docs()[10])

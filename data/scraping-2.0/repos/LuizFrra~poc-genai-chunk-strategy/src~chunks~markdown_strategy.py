from langchain.text_splitter import MarkdownHeaderTextSplitter
from chunks.chunk_strategy import ChunckStrategy

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

class MarkdownStrategy(ChunckStrategy):
    def split(self, text):
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)
        texts = [x.page_content + "\n\n ----" for x in md_header_splits]
        return texts
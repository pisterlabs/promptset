# Here we want to utilize Langchain MarkdownTextSplitter to split the markdown text into chunks
from langchain.text_splitter import MarkdownHeaderTextSplitter
import os

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

# Get each markdown file from the text directory
def get_markdown_files():
    markdown_files = []
    for root, dirs, files in os.walk("text"):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

# Split the markdown files into chunks
def split_markdown_files(markdown_files):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_docs = []
    for file in markdown_files:
        # Read the markdown file
        with open(file, "r") as f:
            markdown_text = f.read()
            md_header_splits = markdown_splitter.split_text(markdown_text)
            split_docs.append(md_header_splits)
    return split_docs

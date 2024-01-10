# Databricks notebook source
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# COMMAND ----------

chunk_size =26
chunk_overlap = 4

# COMMAND ----------

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# COMMAND ----------

"""Why doesn't this split the string below?"""

# COMMAND ----------

text1 = 'abcdefghijklmnopqrstuvwxyz'

# COMMAND ----------

r_splitter.split_text(text1)

# COMMAND ----------

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'

# COMMAND ----------

r_splitter.split_text(text2)

# COMMAND ----------

"""Ok, this splits the string but we have an overlap specified as 5, but it looks like 3? (try an even number)"""

# COMMAND ----------

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

# COMMAND ----------

r_splitter.split_text(text3)

# COMMAND ----------

c_splitter.split_text(text3)

# COMMAND ----------

c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' '
)
c_splitter.split_text(text3)

# COMMAND ----------

"""Recursive splitting details
RecursiveCharacterTextSplitter is recommended for generic text."""

# COMMAND ----------

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

# COMMAND ----------

len(some_text)

# COMMAND ----------

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

# COMMAND ----------

c_splitter.split_text(some_text)

# COMMAND ----------

r_splitter.split_text(some_text)

# COMMAND ----------

"""Let's reduce the chunk size a bit and add a period to our separators:"""

# COMMAND ----------

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)

# COMMAND ----------

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

# COMMAND ----------

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

# COMMAND ----------

docs = text_splitter.split_documents(pages)

# COMMAND ----------

len(docs)

# COMMAND ----------

len(pages)

# COMMAND ----------

from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()

# COMMAND ----------

docs = text_splitter.split_documents(notion_db)

# COMMAND ----------

len(notion_db)

# COMMAND ----------

len(docs)

# COMMAND ----------

"""Token splitting
We can also split on token count explicity, if we want.

This can be useful because LLMs often have context windows designated in tokens.

Tokens are often ~4 characters."""

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter

# COMMAND ----------

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

# COMMAND ----------

text1 = "foo bar bazzyfoo"

# COMMAND ----------

text_splitter.split_text(text1)

# COMMAND ----------

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

# COMMAND ----------

docs = text_splitter.split_documents(pages)

# COMMAND ----------

docs[0]

# COMMAND ----------

pages[0].metadata

# COMMAND ----------

Context aware splitting
Chunking aims to keep text with common context together.

A text splitting often uses sentences or other delimiters to keep related text together but many documents (such as Markdown) have structure (headers) that can be explicitly used in splitting.

We can use MarkdownHeaderTextSplitter to preserve header metadata in our chunks, as show below.

# COMMAND ----------

from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

# COMMAND ----------

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

# COMMAND ----------

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# COMMAND ----------

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# COMMAND ----------

md_header_splits[0]

# COMMAND ----------

md_header_splits[1]

# COMMAND ----------

"""Try on a real Markdown file, like a Notion database."""

# COMMAND ----------

"""loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])"""

# COMMAND ----------

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# COMMAND ----------

md_header_splits = markdown_splitter.split_text(txt)

# COMMAND ----------

md_header_splits[0]

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



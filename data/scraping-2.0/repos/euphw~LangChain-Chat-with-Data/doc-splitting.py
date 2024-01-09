import os
import openai
import sys
sys.path.append('../..')

import constants

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["OPENAI_API_KEY"] = constants.APIKEY

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
text1 = 'abcdefghijklmnopqrstuvwxyz'
#['abcdefghijklmnopqrstuvwxyz']
r_splitter.split_text(text1)
text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
#['abcdefghijklmnopqrstuvwxyz', 'wxyzabcdefg']
r_splitter.split_text(text2)

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
# ['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']
r_splitter.split_text(text3)
# ['a b c d e f g h i j k l m n o p q r s t u v w x y z']
c_splitter.split_text(text3)


#Recursive splitting details
#RecursiveCharacterTextSplitter is recommended for generic text.
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
len(some_text) # 496
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
c_splitter.split_text(some_text)
#['When writing documents, writers will use document structure to group content. This can convey to the reader, which idea\'s are related. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this string. Sentences have a period at the end, but also,', 'have a space.and words are separated by space.']
r_splitter.split_text(some_text)
#["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.",'Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this string. Sentences have a period at the end, but also, have a space.and words are separated by space.']

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)
# ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related",
#  '. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.',
#  'Paragraphs are often delimited with a carriage return or two carriage returns',
#  '. Carriage returns are the "backslash n" you see embedded in this string',
#  '. Sentences have a period at the end, but also, have a space.and words are separated by space.']
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)
# ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related.",
#  'For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.',
#  'Paragraphs are often delimited with a carriage return or two carriage returns.',
#  'Carriage returns are the "backslash n" you see embedded in this string.',
#  'Sentences have a period at the end, but also, have a space.and words are separated by space.']
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs\LLXZ.pdf")
pages = loader.load()
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)
len(docs)
len(pages)
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()
docs = text_splitter.split_documents(notion_db)
len(notion_db)
len(docs)

#Token splitting
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(
    chunk_size=1, chunk_overlap=0
)
text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
docs[0]
pages[0].metadata

#Context aware splitting
# Chunking aims to keep text with common context together.
# A text splitting often uses sentences or other delimiters to keep related text together but many documents (such as Markdown) have structure (headers) that can be explicitly used in splitting.
# We can use MarkdownHeaderTextSplitter to preserve header metadata in our chunks, as show below.
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits[0]
# Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Title', 'Header 2': 'Chapter 1'})
md_header_splits[1]
# Document(page_content='Hi this is Lance', metadata={'Header 1': 'Title', 'Header 2': 'Chapter 1', 'Header 3': 'Section'})

# Try on a real Markdown file, like a Notion database.
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(txt)
md_header_splits[0]

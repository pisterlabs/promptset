import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

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

# Print chunk size and overlap
print(f'Chunk size: {chunk_size}')
print(f'Chunk overlap: {chunk_overlap}')

text1 = 'abcdefghijklmnopqrstuvwxyz'
# Print text 1 with a comment to separate it from text 2
print(f'This is text1: {text1}')
print('---')
# Print text 1 recursively split
print('Recursive splitter output for text1:')
print(r_splitter.split_text(text1))
print()
print()

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print(f'This is text2: {text2}')
print('Recursive splitter output for text2:')
print(r_splitter.split_text(text2))

print('---')
text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print(f'This is text3: {text3}')


print('Recursive splitter output for text3:')
print(r_splitter.split_text(text3))

print('Character splitter output for text3:')
print(c_splitter.split_text(text3))

print('---')
c_splitter_2 = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' '
)
print('Added separator /n=' ' to character splitter output for text3:')
print('Character splitter with separator output for text3:')
print(c_splitter_2.split_text(text3))


# Recursive splitter detail
print('---')
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""] # We want to split on newlines, periods, and spaces
)

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/Assura-Basis_CGA_LAMal_2024_F.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)

print(docs[0].page_content[:1000])
print(f' Len doc: {len(docs)} vs len pages: {len(pages)}')


print('---')
# Token splitting
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

text1 = "foo bar bazzyfoo"
print(f'This is text1: {text1}')

print('Token splitter output for text1, chunk size 1 and overlap 0:')
print(text_splitter.split_text(text1))

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
print('Token splitter output for text1, chunk size 10 and overlap 0:')
print(text_splitter.split_text(text1))


print('---')
# Context aware splitting
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n
## Chapter 2\n\n \
Hi this is Molly"""
print(markdown_document)


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
print(md_header_splits)

# Notion document aware splitting
print('---')
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

print(md_header_splits[0])

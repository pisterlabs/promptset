# This is a long document we can split up.
with open('../../../yourdoc.txt') as f:
    file = f.read()

#Recursive chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len)

texts = text_splitter.create_documents([file])

print(texts[0])
print(texts[1])

#Split a document by sentence
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp('This is the first sentence. This is the second sentence.')
for sent in doc.sents:
    print(sent)

#Splitting a document by a fixed token size
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(file)
print(texts[0])

#Splitting a document by the document's inherent structure
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits


#Output
#    [Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}),
#     Document(page_content='Hi this is Lance', metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}),
#     Document(page_content='Hi this is Molly', metadata={'Header 1': 'Foo', 'Header 2': 'Baz'})]`
import openai
import pinecone
from time import sleep
import uuid
import os
from PineconeDB import Database
import PyPDF2
from uuid import uuid4
from tqdm.auto import tqdm
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import Markdown

pdf_paths = \
[r'ASHA Manuals/ASHA_Handbook-Mobilizing_for_Action_on_Violence_against_Women_English.pdf', 
r'ASHA Manuals/book-no-1.pdf', 
r'ASHA Manuals/book-no-2.pdf',
r'ASHA Manuals/book-no-3.pdf',
r'ASHA Manuals/book-no-4.pdf',
r'ASHA Manuals/book-no-5.pdf',
r'ASHA Manuals/book-no-6.pdf',
r'ASHA Manuals/book-no-7.pdf']

# read every pdf and get chunks
chunks = []
for pdf_path in pdf_paths:
    pdf_file = open(pdf_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_count = len(pdf_reader.pages)
    print("Page count:", page_count)

    def pdf_len(text):
        return len(text)

    print("Reading...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=pdf_len,
        separators=["\n\n", "\n", " ", ""]
    )

    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        text = page.extract_text()
        texts = text_splitter.split_text(text)
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[j],
            'chunk': j,
            'page': i,
            'pdf': pdf_path,
        } for j in range(len(texts))])

# print(len(chunks))


embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

INDEX_NAME = "test"

PINECONE_ENVIRONMENT = "gcp-starter"
EMBED_MODEL = "text-embedding-ada-002"

db = Database(INDEX_NAME, PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBED_MODEL)

index = pinecone.Index(INDEX_NAME)
index.describe_index_stats()

# print(chunks[:5])

# for chunk in chunks:
#     text = chunk['text']
#     # print(text)
#     db.insert(text)

# db.insert("How many times have you gone to the bathroom?")
# db.insert("When was the last time you went to the bathroom?")
# db.insert("Was there any blood in the stool?")
# db.insert("Do you have any other conditions that I should be aware of?")
# db.insert("What is your name?")

query_message = "A female child, age 5, presents with diarrhea."
retrieval_results = db.retrieve(query_message)

print("Retrieval Results:")
print(retrieval_results)
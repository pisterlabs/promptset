import sys
import argparse
import os
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Specify the default values
DEFAULT_OPENAI_API_KEY = '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'

# Parse command line arguments
args = dict(arg.split('=') for arg in sys.argv[1:])
directory = args.get("--faq-dir", "CustomerSupportFAQs")
faq_embedding_store = args.get("--faq-embedding-store", 'faq-embeddings-store')
openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

docs = []
# Traverse the directory and get each file name
for filename in os.listdir(directory):
    filePath = os.path.join(directory, filename)
    if os.path.isfile(filePath):
        print(filePath)
        loader = TextLoader(filePath)
        docs.extend(loader.load())

print(f"{len(docs)} documents with a total {sum([len(x.page_content) for x in docs])} characters")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)

for doc in docs:
    doc.page_content = doc.page_content.replace("\n", " ")
docs = text_splitter.split_documents(docs)
print(f"After split: {len(docs)} documents with a total {sum([len(x.page_content) for x in docs])} characters")

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)

# Example search
query = 'How do I request a product return?'
found_docs = docsearch.similarity_search(query)
print(f'query="{query}" and result: {found_docs}')

# Save the vectorstore to a local directory
docsearch.save_local(faq_embedding_store)

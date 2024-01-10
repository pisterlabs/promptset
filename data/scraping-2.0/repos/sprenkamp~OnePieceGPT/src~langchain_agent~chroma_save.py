from helper.clean import clean_webtext_using_GPT
from helper.split import split_text
from helper.scrape import scrape_webpage 

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from tqdm import tqdm

persistent_client = chromadb.PersistentClient(path="data/chroma/one_piece/")

with open("data/queries/websites_to_srape.txt") as f:
    URLs = f.readlines()
    URLs = [x.strip() for x in URLs]
    URLs = [x for x in URLs if x.startswith("http")]

for URL in tqdm(URLs, desc="adding documents from URLs to chromadb"):
    initial_doc = scrape_webpage(URL)
    docs = split_text(initial_doc)
    for i, doc in tqdm(enumerate(docs), desc=URL):
        doc.page_content = doc.page_content.replace("\n", " ")
        doc.page_content = clean_webtext_using_GPT(doc.page_content)
        docs[i].page_content = doc.page_content
    docs = [doc for doc in docs if doc.page_content != "NO_INFORMATION"]
    vectordb = Chroma.from_documents(
        collection_name="one_piece",
        client=persistent_client,
        documents=docs,
        embedding=OpenAIEmbeddings(),
        )
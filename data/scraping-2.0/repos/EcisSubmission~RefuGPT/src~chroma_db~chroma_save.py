import argparse
import sys
sys.path.append("src/langchain_agent/helper/")
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from clean import clean_webtext_using_GPT
from split import create_documents_from_texts, split_documents
from scrape import scrape_webpage 


def process_single_url(URL):
    initial_doc = scrape_webpage(URL)
    docs = split_documents(initial_doc)
    for i, doc in enumerate(docs):
        doc.page_content = doc.page_content.replace("\n", " ")
        doc.page_content = clean_webtext_using_GPT(doc.page_content)
        if "NO_INFORMATION" in doc.page_content:
            continue
        doc.page_content = doc.page_content + "\n source URL: " + URL
        docs[i].page_content = doc.page_content

    vectordb = Chroma.from_documents(
        collection_name="swiss_refugee_info_source",
        client=persistent_client,
        documents=docs,
        embedding=OpenAIEmbeddings(),
    )

def save_data_swiss_official_from_URL():    
    with open("data/chroma/input/web_queries/websites_to_srape.txt") as f:
        URLs = f.readlines()
        URLs = [x.strip() for x in URLs]
        URLs = [x for x in URLs if x.startswith("http")]
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_url, URLs), total=len(URLs)))

def process_multiple_rows(rows):
    metadata_list = []
    texts = []
    for row in rows:
        # Create metadata dictionary for each row
        metadata = {'date': row['messageDatetime']}
        metadata_list.append(metadata)
        texts.append(row['messageText'])

    docs = create_documents_from_texts(texts, metadata_list)
    
    Chroma.from_documents(
        collection_name="community_refugee_info_extensive",
        client=persistent_client,
        documents=docs, 
        embedding=OpenAIEmbeddings(),
    )
    

def save_data_telegram_community_data():
    df = pd.read_csv("data/chroma/input/df_telegram_for_chroma.csv")
    df = df[df.messageText.str.len()>100] # Remove short messages
    # Create batches of rows
    batch_size = 10  
    rows = [row for _, row in df.iterrows()]
    row_batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    for batch in tqdm(row_batches):
        process_multiple_rows(batch)
    # with ThreadPoolExecutor(max_workers=5) as executor: TODO fix issue with multithreading
    #     list(tqdm(executor.map(process_multiple_rows, row_batches), total=len(row_batches), desc=f"Processing rows in batches of {batch_size}"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose which method to run.')
    parser.add_argument('--method', choices=['official', 'community'], help='Method to run: official for Swiss Official Data, telegram for Telegram Community Data')
    args = parser.parse_args()
    
    if args.method == 'official':
        persistent_client = chromadb.PersistentClient(path="data/chroma/swiss_refugee_info_source/")
        save_data_swiss_official_from_URL()
    elif args.method == 'community':
        persistent_client = chromadb.PersistentClient(path="data/chroma/community_refugee_info/")
        save_data_telegram_community_data()

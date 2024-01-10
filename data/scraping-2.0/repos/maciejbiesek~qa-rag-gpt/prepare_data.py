import os
import pickle
from typing import List, Dict

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from utils import load_config, save_dataframe


def download_data(number_of_document: int) -> pd.DataFrame:
    dataset = load_dataset("WiktorS/polish-news")
    df = pd.DataFrame(dataset['train'][:number_of_document])
    df['text'] = df.apply(lambda x: f"{x['title'].strip()} {x['headline'].strip()} {x['content'].strip()}", axis=1)
    return df[['link', 'text']]


def embedd_documents(docs: List[str], metadata: List[Dict], vectorstore_filepath: str):
    # Creating vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings, metadatas=metadata)

    with open(vectorstore_filepath, 'wb') as f:
        pickle.dump(vectorstore, f)


def process_data(data: List[Dict[str, str]], vectorstore_filepath: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    # Split documents into chunks
    processed_docs, metadata = [], []
    for doc in tqdm(data):
        chunks = text_splitter.split_text(doc['text'])
        processed_docs.extend(chunks)
        metadata.extend([{"source": doc['link']}] * len(chunks))

    embedd_documents(processed_docs, metadata, vectorstore_filepath)


def main():
    config_data = load_config("DATASET")
    load_dotenv()

    data = download_data(config_data.getint("NUMBER_OF_DOCS"))
    save_dataframe(data, os.path.join(config_data.get("DATA_PATH"), config_data.get("DATA_FILENAME")))
    data = data.to_dict(orient='records')
    process_data(data, os.path.join(config_data.get("DATA_PATH"), config_data.get("VECTORSTORE_FILENAME")))


if __name__ == "__main__":
    main()

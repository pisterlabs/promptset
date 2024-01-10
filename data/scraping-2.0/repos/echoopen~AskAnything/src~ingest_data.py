# Using langchain, ingest data from a website to vector store
import os
import re
import argparse
import traceback
import configparser
from tqdm import tqdm
from app_config import *
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))


def main(args: argparse.Namespace) -> dict:
    res = {"status": 0, "message": "Success"}

    # Sanity check inputs
    config_fn = os.path.join(FILE_ROOT, args.config)
    if not os.path.exists(config_fn):
        res["status"] = 2
        res["message"] = f"Config file {config_fn} does not exist"
        return res

    # Load the config file
    try:
        site_config = configparser.ConfigParser()
        site_config.read(config_fn)
        site_section = site_config[args.site]

        index_url = site_section["index"]
        url_filters = site_section["url_filters"].split(";")
        url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in url_filters]
        debug_url_filters = site_section["debug_url_filters"].split(";")
        debug_url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in debug_url_filters]
        custom_separators = site_section["custom_separators"].split(";")
        negative_text_page = site_section["negative_text_page"].split(";")
        negative_text_chunk = site_section["negative_text_chunk"].split(";")
        min_chunk_length = int(site_section["min_chunk_length"])

        # Remove any escaped characters from the separators and filters
        for lst in [
            custom_separators,
            negative_text_page,
            negative_text_chunk
        ]:
            for i in range(len(lst)):
                lst[i] = lst[i].replace("\\n", "\n").replace("\\r", "\r")

        if args.debug:
            print(f"index_url = {index_url}")
            print(f"url_filters = {url_filters}")
            print("Replacing the url_filters with one specific for debug purposes")
            url_filters = debug_url_filters
            print(f"Adjusted url_filters = {url_filters}")
            print(f"custom_separators = {custom_separators}")
            print(f"negative_text_page = {negative_text_page}")
            print(f"negative_text_chunk = {negative_text_chunk}")
            print(f"min_chunk_length = {min_chunk_length}")

    except:
        res["status"] = 2
        res["message"] = f"Error reading config file {config_fn}: {traceback.format_exc()}"
        return res

    # Initialize all needed objects

    # Sitemap loader
    loader = SitemapLoader(index_url, url_filters)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)

    # Load the sitemap
    try:
        docs = loader.load()
    except:
        res["status"] = 2
        res["message"] = f"Error loading sitemap {index_url}: {traceback.format_exc()}"
        return res

    all_texts = []
    post_filter_docs = 0
    for doc in tqdm(docs, desc="Filtering documents", ascii=True):
        # Skip entire page if it contains any negative_text_page items
        if any([re.search(filter, doc.page_content) for filter in negative_text_page]):
            continue

        # Split the document page_content into text chunks based on the custom separators using re
        chunks = re.split("|".join(custom_separators), doc.page_content)

        # Perform sanity check on any negative filters, then reduce any length of \n to a single \n in each chunk
        final_chunks = []
        for chunk in chunks:
            if not any([re.search(filter, chunk) for filter in negative_text_chunk]):
                final_chunks.append(re.sub("\n+", "\n", chunk))

        # Copy the doc.metadata into a list of metadata the length of chunks list
        metadatas = [doc.metadata] * len(final_chunks)

        texts = text_splitter.create_documents(final_chunks, metadatas)
        for text in texts:
            # Filter by minimum length, or else too short and uninformative
            if len(text.page_content.strip()) >= min_chunk_length:
                all_texts.append(text)

        # Increase number of documents that passed the filter
        post_filter_docs += 1

    print(f"Number of documents after filtering: {post_filter_docs}")
    print(f"Number of text chunks after filtering: {len(all_texts)}")

    # Embedding model
    embedding = OpenAIEmbeddings()

    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = os.path.join(FILE_ROOT, CHROMA_DB_DIR, args.site.replace(".", "_")).rstrip("/")
    vector_db = Chroma.from_documents(documents=all_texts, embedding=embedding, persist_directory=persist_directory)

    # Save the vector store
    try:
        vector_db.persist()
        vector_db = None
    except:
        res["status"] = 2
        res["message"] = f"Error persisting vector store: {traceback.format_exc()}"
        return res

    # Compress the vector store into a tar.gz file of the same name
    tar_cmd = f"tar -czvf {persist_directory}.tar.gz -C {os.path.dirname(persist_directory)} {os.path.basename(persist_directory)}"
    try:
        os.system(tar_cmd)
    except:
        res["status"] = 2
        res["message"] = f"Error compressing vector store: {traceback.format_exc()}"
        return res

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("--site", type=str, required=True, help="Site to ingest (must be a section in the config file!)")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="cfg/default.cfg")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    run_res = main(args)

    if run_res["status"] != 0:
        print(run_res["message"])
        exit(run_res["status"])

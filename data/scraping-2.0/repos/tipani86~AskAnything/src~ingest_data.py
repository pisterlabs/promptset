# Using langchain, ingest data from a website to vector store
import os
import re
import argparse
import traceback
import configparser
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
from app_config import *
from loguru import logger
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.document_loaders import DataFrameLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_ROOT = Path(__file__).parent
chunk_size = 2000
chunk_overlap = 200

def main(args: argparse.Namespace) -> tuple[int, str]:
    status = 0
    message = "Success"
    if args.dry_run:
        logger.warning("Dry run mode enabled! (No vectorization or database save)")
    else:
        # Check if the OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            status = 2
            message = "OpenAI API key not set! Please set the OPENAI_API_KEY environment variable to your OpenAI API key"
            return status, message
    if args.debug:
        logger.warning("Debug mode enabled! (Depending on the config, the behavior may change)")

    # Sanity check inputs
    config_fn = FILE_ROOT / args.config
    if not config_fn.exists():
        status = 2
        message = f"Config file {config_fn} does not exist"
        return status, message

    # Load the config file
    config_basename = config_fn.stem
    config = configparser.ConfigParser()
    config.read(config_fn)

    all_texts = []
        
    if "sitemap" in config.sections():
        try:
            section = config["sitemap"]

            index_url = section["index"]
            url_filters = section["url_filters"].split(";")
            url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in url_filters]
            debug_url_filters = section["debug_url_filters"].split(";")
            debug_url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in debug_url_filters]
            custom_separators = section["custom_separators"].split(";")
            negative_text_page = section["negative_text_page"].split(";")
            negative_text_chunk = section["negative_text_chunk"].split(";")
            min_chunk_length = int(section["min_chunk_length"])
            chunk_ratio = float(section.get("chunk_ratio", 1.0))

            # Remove any escaped characters from the separators and filters
            for lst in [
                custom_separators,
                negative_text_page,
                negative_text_chunk
            ]:
                for i in range(len(lst)):
                    lst[i] = lst[i].replace("\\n", "\n").replace("\\r", "\r")

            if args.debug:
                logger.debug(f"Config type: {section}")
                logger.debug(f"index_url = {index_url}")
                logger.debug(f"url_filters = {url_filters}")
                logger.debug("Replacing the url_filters with one specific for debug purposes")
                url_filters = debug_url_filters
                logger.debug(f"Adjusted url_filters = {url_filters}")
                logger.debug(f"custom_separators = {custom_separators}")
                logger.debug(f"negative_text_page = {negative_text_page}")
                logger.debug(f"negative_text_chunk = {negative_text_chunk}")
                logger.debug(f"min_chunk_length = {min_chunk_length}")

        except:
            status = 2
            message = f"Error reading config file {config_fn}: {traceback.format_exc()}"
            return status, message

        # Initialize all needed objects

        # Sitemap loader
        loader = SitemapLoader(index_url, url_filters)

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_ratio * chunk_size),
            chunk_overlap=int(chunk_ratio * chunk_overlap),
        )

        # Load the sitemap
        try:
            docs = loader.load()
        except:
            status = 2
            message = f"Error loading sitemap {index_url}: {traceback.format_exc()}"
            return status, message

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

        logger.info(f"Number of documents after filtering: {post_filter_docs}")
        logger.info(f"Number of text chunks after filtering: {len(all_texts)}")


    # TODO very much in-progress, not production-ready
    if "excel" in config.sections():
        section = config["excel"]
        input_fn = section["input_fn"]

        df = pd.read_excel(input_fn)
        loader = DataFrameLoader(df, page_content_column="Product Details")
        docs = loader.load()

        all_texts = docs

    # TODO: Try unstructured method to get page numbers etc.
    if "pdf" in config.sections():
        try:
            section = config["pdf"]
            input_fn = section["input_fn"]
            chunk_ratio = float(section.get("chunk_ratio", 1.0))
        except:
            status = 2
            message = f"Error reading config file {config_fn}: {traceback.format_exc()}"
            return status, message
        
        if args.debug:
            logger.debug(f"Config type: {section}")
            logger.debug(f"input_fn = {input_fn}")

        # Initialize all needed objects

        # PDF loader
        loader = PyMuPDFLoader(input_fn)

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_ratio * chunk_size),
            chunk_overlap=int(chunk_ratio * chunk_overlap),
        )

        # Load the PDF
        try:
            docs = loader.load()
        except:
            status = 2
            message = f"Error loading PDF {input_fn}: {traceback.format_exc()}" 
            return status, message
        
        # Save the input file's basename as the docs metadata source
        for doc in docs:
            doc.metadata["source"] = os.path.basename(input_fn)

        all_texts = text_splitter.split_documents(docs)
        # Fix page numbers (because they are 0-indexed)
        for text in all_texts:
            text.metadata["page"] += 1

        if args.debug:
            # Print the first 5 text entries
            for i, text in enumerate(all_texts[:5]):
                logger.debug(f"Debug printing text {i+1}")
                print(text.page_content)

        logger.info(f"Number of documents: {len(docs)}")
        logger.info(f"Number of text chunks: {len(all_texts)}")

    if "site_excel" in config.sections():
        try:
            section = config["site_excel"]
            input_fn = section["input_fn"]
            chunk_ratio = float(section.get("chunk_ratio", 1.0))
        except:
            status = 2
            message = f"Error reading config file {config_fn}: {traceback.format_exc()}"
            return status, message

        # Load the excel files into a dataframe
        df = pd.read_excel(input_fn, engine="openpyxl")

        # Group the dataframe by site column
        grps = df.groupby('site')

        all_texts = []
        for site, gdf in grps:
            logger.info(f"Processing site {site}...")

            # Get site-specific configs
            try:
                site_config = config[site]
            except:
                status = 2
                message = f"Error searching {site} in config file {config_fn}: {traceback.format_exc()}"
                return status, message

            start_after = site_config.get("start_after", "")
            stop_after = site_config.get("stop_after", "")

            urls = gdf['url'].tolist()
            loader = UnstructuredURLLoader(urls, mode="elements", headers={"User-Agent": "Mozilla/5.0"}, show_progress_bar=True)
            docs = loader.load()

            # Create a url to document text lookup dict
            url_doc_contents_map = {}
            for doc in docs:
                # Skip all non-text document parts
                if doc.metadata["category"] != "Title":
                    continue
                url = doc.metadata["url"]
                if url not in url_doc_contents_map:
                    url_doc_contents_map[url] = []
                url_doc_contents_map[url].append(doc.page_content)

            # Post-process the documents and add the results to the final_documents list
            for url, texts in url_doc_contents_map.items():
                # Make a single text chunk from the entire document by joining each text element with a paragraph break
                joined_text = "\n\n".join(texts)

                # If keyword argument start_after is set, cut off any text up to the first occurrence of the keyword
                if len(start_after) > 0:
                    joined_text = joined_text.split(start_after, 1)[-1]

                # If keyword argument stop_after is set, cut off any text after the first occurrence of the keyword
                if len(stop_after) > 0:
                    joined_text = joined_text.split(stop_after, 1)[0]

                # Use text splitter to split the text into sentences
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=int(chunk_ratio * chunk_size),
                    chunk_overlap=int(chunk_ratio * chunk_overlap),
                )
                split_texts = text_splitter.split_text(joined_text)

                # Create a document for each split text
                metadatas = [{"url": url}] * len(split_texts)
                all_texts.extend(text_splitter.create_documents(split_texts, metadatas))

        logger.info(f"Generated {len(all_texts)} text chunks from {len(df)} urls")

    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = str(FILE_ROOT / CHROMA_DB_DIR / config_basename.replace(".", "_")).rstrip("/")
    if args.debug:
        logger.debug(f"persist_directory = {persist_directory}")
    if args.dry_run:
        logger.warning("Stopping processing due to dry_run mode!")
        return status, message
    
    # Embedding model
    embedding = OpenAIEmbeddings()

    vector_db = Chroma.from_documents(documents=all_texts, embedding=embedding, persist_directory=persist_directory)

    # Save the vector store
    try:
        vector_db.persist()
        vector_db = None
    except:
        status = 2
        message = f"Error persisting vector store: {traceback.format_exc()}"
        return status, message

    # Compress the vector store into a tar.gz file of the same name
    tar_cmd = f"tar -czvf {persist_directory}.tar.gz -C {str(Path(persist_directory).parent)} {str(Path(persist_directory).name)}"
    if args.debug:
        logger.debug(f"tar_cmd = {tar_cmd}")
    
    run_res = os.system(tar_cmd)
    if run_res != 0:
        status = 2
        message = f"Error running tar command: {tar_cmd}"
        return status, message

    return status, message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dry_run", action="store_true", help="Enable dry run mode (do not vectorize or save to database)")
    args = parser.parse_args()

    status, message = main(args)

    if status != 0:
        logger.error(message)
        exit(status)

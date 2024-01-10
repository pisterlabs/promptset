# File: scraper.py
# Description: This script defines functions for scraping and processing scientific papers from bioRxiv,
# extracting text and embeddings, and storing the information in a custom database.
# It also performs a keyword search on the obtained data.

# Importing necessary libraries
import os
import pandas as pd
import PyPDF2
import argparse, datetime
from paperscraper.pdf import save_pdf
from paperscraper.get_dumps import biorxiv
from paperscraper.xrxiv.xrxiv_query import XRXivQuery

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
import PyPDF2

from VectorDatabase import Lantern, Fragment, Publication


# OpenAI Setup
# openai.api_key = os.getenv(openai_api_key)
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


"""
Scrapes papers from bioRxiv between the specified dates and saves the metadata in a JSON file.

:param start: Start date for the scraping (format: "YYYY-MM-DD").
:param end: End date for the scraping (format: "YYYY-MM-DD").
:param out_file: Output file to save the metadata in JSON Lines format.
:return: None
"""
def scrapeBiorxiv(start, end, out_file):
    filepath = out_file
    biorxiv(begin_date=start, end_date=end, save_path=out_file)
    retreiveTextFromPdf(filepath)

"""
Retrieves text embeddings from a given text file using OpenAI's language model.

:param fname: Path to the input text file.
:return: A tuple containing text embeddings and the OpenAIEmbeddings instance.
"""
def get_embeddings(fname):
    loader = TextLoader(fname)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator=".", chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    emb = OpenAIEmbeddings()
    input_texts = [d.page_content for d in docs]

    input_embeddings = emb.embed_documents(input_texts)
    text_embeddings = list(zip(input_texts, input_embeddings))
    return text_embeddings, emb


"""
Retrieves text from PDF files, extracts embeddings, and stores information in a custom database.

:param inp_file: Path to the input JSON file containing paper metadata.
:return: None
"""
def retreiveTextFromPdf(inp_file):

    json = pd.read_json(path_or_buf=inp_file, lines=True)
    lantern = Lantern()

    for n, doi in enumerate(json['doi']):

        paper_data = {'doi': doi}
        doi = doi.replace("/", "-")

        if lantern.publicationExists(doi):
            continue

        pdf_dir = './papers/'
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

        pdfsavefile = './papers/' + doi + '.pdf'
        save_pdf(paper_data, filepath=pdfsavefile)

        # creating a pdf reader object
        reader = PyPDF2.PdfReader(pdfsavefile)
        save_txt_path = 'scrapped_txts/'
        if not os.path.exists(save_txt_path):
            os.mkdir(save_txt_path)
        extract_text = ''
        for page in reader.pages:
            extract_text += page.extract_text()

        txt_file = str('{}.txt'.format(doi))
        with open(save_txt_path + txt_file, 'w') as file:
            file.write(extract_text)

        txt_embs, emb = get_embeddings(save_txt_path + txt_file)

        fragments = []
        for txt, embs in txt_embs:
            fragment = Fragment(doi, 'methods', txt, embs)
            fragments.append(fragment)

        title = ""
        pmc = ""
        pubmed = ""

        publication = Publication(doi, title, pmc, pubmed, doi)

        lantern.insertEmbeddings(fragments)
        lantern.insertPublication(publication)

        os.remove(pdfsavefile)


if __name__ == "__main__":
    # Adding command line arguments for start_date and end_date with default values as the current date
    parser = argparse.ArgumentParser(description="Scrape and process scientific papers from bioRxiv.")
    parser.add_argument("--start-date", default=str(datetime.date.today()), help="Start date for the scraping (format: 'YYYY-MM-DD').")
    parser.add_argument("--end-date", default=str(datetime.date.today()), help="End date for the scraping (format: 'YYYY-MM-DD').")
    parser.add_argument("--outfile", default="bio.jsonl", help="Output file to save the metadata in JSON Lines format.")
    args = parser.parse_args()

    # Calling the scrapeBiorxiv function with command line arguments
    scrapeBiorxiv(args.start_date, args.end_date, args.out_file)

    # Additional code for keyword search if needed
    querier = XRXivQuery(args.out_file)
    biology = ['Bioinformatics', 'Molecular Biology', 'Bioengineering', 'Biochemistry']
    query = [biology]
    querier.search_keywords(query, output_filepath='bio_key.jsonl')
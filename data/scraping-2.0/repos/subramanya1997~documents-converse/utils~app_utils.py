"""
   Created on Thu Jun 08 2023
   Copyright (c) 2023 Subramanya N
"""
import os
import logging
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter
from zipfile import ZipFile
from uuid import uuid4
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import convertPDFToText
from llm_openai import create_embeddings
from pineconesearch import pinecone_search

logger = logging.getLogger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")


def validate_zip_file(filepath):
    logger.debug(f"Validating zip file: {filepath}")
    if not os.path.exists(filepath):
        return False
    if not filepath.endswith(".zip"):
        return False
    # extract the zip file and check there is at least one file with a valid extension
    count = 0
    with ZipFile(filepath, "r") as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Check if the extracted files are only PDFs and images
        allowed_extensions = {"pdf", "jpg", "jpeg", "png"}
        for filename in listOfFileNames:
            # ignore directories
            if filename.endswith("/"):
                continue
            # ignore files with no extension
            if "." not in filename:
                continue
            # ignore backup files like filename.pdf.bak, __MACOSX
            if filename.split(".")[-1] in {"bak", "DS_Store", "MACOSX"}:
                continue
            if filename.split(".")[-1].strip() in allowed_extensions:
                count += 1
    if count == 0:
        return False
    return True


def split_pdf(input_path, output_prefix, max_pages):
    max_pages = int(max_pages)
    pdf_chunks_paths = []
    with open(input_path, "rb") as input_file:
        pdf = PdfReader(input_file)
        total_pages = len(pdf.pages)
        num_chunks = (total_pages + max_pages - 1) // max_pages
        for i in range(num_chunks):
            s_page = i * max_pages
            e_page = min((i + 1) * max_pages, total_pages)

            output_path = f"{output_prefix}$${i+1}.pdf"
            with open(output_path, "wb") as output_file:
                output_pdf = PdfWriter()
                for page in range(s_page, e_page):
                    output_pdf.add_page(pdf.pages[page])

                output_pdf.write(output_file)
            pdf_chunks_paths.append(output_path)
    return pdf_chunks_paths


def process_zip(filepath):
    logger.debug(f"Processing zip file: {filepath}")
    if not os.path.exists(filepath):
        return False
    if not filepath.endswith(".zip"):
        return False
    # get all pdfs and images from the zip file which
    allowed_extensions = {"pdf", "jpg", "jpeg", "png"}
    # create a temporary directory to store the extracted files
    temp_dir = os.path.join(
        os.path.dirname(filepath), f"temp_{os.path.basename(filepath)}"
    )
    with ZipFile(filepath, "r") as zipObj:
        listOfFileNames = zipObj.namelist()
        for filename in listOfFileNames:
            # ignore directories
            if filename.endswith("/"):
                continue
            # ignore files with no extension
            if "." not in filename:
                continue
            # ignore backup files like filename.pdf.bak, __MACOSX
            if filename.split(".")[-1].strip() in {"bak", "DS_Store", "MACOSX"}:
                continue
            # ignore directories starting with __ (like __MACOSX)
            c_flag = False
            for dir in filename.split("/")[:-1]:
                if dir.strip().startswith("__"):
                    c_flag = True
                    break
            if c_flag:
                continue
            if filename.split(".")[-1].strip() in allowed_extensions:
                zipObj.extract(filename, temp_dir)

    extracted_text = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            document = {
                "DocumentType": None,
                "DocumentName": file,
                "Content": None,
            }
            if file.split(".")[-1].strip() in allowed_extensions:
                _filepath = os.path.join(root, file)
                if file.split(".")[-1].strip() == "pdf":
                    logger.debug(f"Processing PDF: {_filepath}")
                    document["DocumentType"] = "pdf"
                    try:
                        with open(_filepath, "rb") as input_file:
                            pdf = PdfReader(input_file)
                            total_pages = len(pdf.pages)
                    except:
                        continue
                    if total_pages > 15:
                        output_prefix = _filepath.split(".pdf")[0]
                        pdf_chunks_paths = split_pdf(
                            _filepath, output_prefix, os.environ["MAX_PDF_PAGES"]
                        )

                        text = ""
                        for pdf_chunk_path in pdf_chunks_paths:
                            text += convertPDFToText(pdf_chunk_path)
                    else:
                        text = convertPDFToText(_filepath)
                    document["Content"] = text
                else:
                    text = convertPDFToText(_filepath)
                    document["DocumentType"] = "image"
                    document["Content"] = text
            if document["Content"] is not None:
                extracted_text.append(document)
    # remove the temporary directory
    os.system(f"rm -rf {temp_dir}")
    os.system(f"rm {filepath}")
    generate_embeddings_upsert(extracted_text)
    return True


def token_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def generate_embeddings_upsert(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for document in documents:
        texts = text_splitter.split_text(document["Content"])
        chunks.extend(
            [
                {
                    "id": str(uuid4()),
                    "text": texts[i],
                    "chunk_index": i,
                    "title": document["DocumentName"],
                    "type": document["DocumentType"],
                }
                for i in range(len(texts))
            ]
        )
    pinecone_search.delete_documents()
    for i in tqdm(range(0, len(chunks), 100)):
        # find end of batch
        i_end = min(len(chunks), i + 100)
        batch = chunks[i:i_end]
        ids_batch = [x["id"] for x in batch]
        texts = [x["text"] for x in batch]
        embeds = create_embeddings(text=texts)
        # cleanup metadata
        meta_batch = [
            {
                "title": x["title"],
                "type": x["type"],
                "text": x["text"],
                "chunk_index": x["chunk_index"],
            }
            for x in batch
        ]
        to_upsert = []
        for id, embed, meta in list(zip(ids_batch, embeds, meta_batch)):
            to_upsert.append(
                {
                    "id": id,
                    "values": embed,
                    "metadata": meta,
                }
            )
        # upsert to Pinecone
        pinecone_search.upsert_documents(to_upsert)

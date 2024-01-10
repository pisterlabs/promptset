import json
import os
import re
import tempfile

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_fixed
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean, group_broken_paragraphs
from unstructured.documents.elements import (
    CompositeElement,
    Footer,
    Header,
    Image,
    NarrativeText,
    Title,
    Table,
)
from unstructured.partition.auto import partition


load_dotenv()



def extract_pages(filename):
    # 获取文件名（不包括扩展名）
    base_name = os.path.splitext(filename)[0]
    # 创建新的文件名
    output_filename = "pdfs/" + base_name + "_new.pdf"

    pdf_reader = PdfReader("datareport/" + filename)
    pdf_writer = PdfWriter()

    total_pages = len(pdf_reader.pages)

    # 提取第一页
    first_page = pdf_reader.pages[0]
    pdf_writer.add_page(first_page)

    # 提取最后两页
    for page_number in range(total_pages - 2, total_pages):
        page = pdf_reader.pages[page_number]
        pdf_writer.add_page(page)

    with open(output_filename, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

    return output_filename


directory = "datareport"

for pdf_name in os.listdir(directory):
    new_pdf_name = extract_pages(pdf_name)
    
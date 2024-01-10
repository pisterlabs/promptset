#  Copyright (c) 2023 Higher Bar AI, PBC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utility functions for reading questionnaire files."""
from typing import Callable
import nltk
import spacy
import os
import csv
import re
import xml.etree.ElementTree as ElementTree
import requests
import pytesseract
from pypdf import PdfReader
from tabula.io import read_pdf
from pdf2image import convert_from_path
from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.schema import Document as SchemaDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader


# initialize global variables
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
LANGCHAIN_SPLITTER_CHUNK_SIZE = 7500
LANGCHAIN_SPLITTER_OVERLAP_SIZE = 500


def clean_whitespace(s: str) -> str:
    """
    Clean and standardize whitespace in a string.

    This includes converting tabs to spaces, trimming leading/trailing whitespace from each line,
    reducing multiple spaces to single ones, and reducing more than three consecutive linebreaks to three.

    :param s: String to clean.
    :type s: str
    :return: Cleaned string.
    :rtype: str
    """

    # convert tabs to spaces
    s = re.sub(r'\t', ' ', s)

    # trim leading and trailing whitespace from each line
    lines = s.split('\n')
    trimmed_lines = [line.strip() for line in lines]
    s = '\n'.join(trimmed_lines)

    # reduce multiple spaces to a single space
    s = re.sub(' +', ' ', s)

    # reduce more than two consecutive linebreaks to two
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s


def split_langchain(content, create_doc: bool = True) -> list:
    """
    Split content into chunks using langchain.

    :param content: Content to split.
    :param create_doc: Flag to indicate whether to create a langchain Document.
    :type create_doc: bool
    :return: List of langchain documents.
    :rtype: list
    """

    # split content into chunks, with overlap to ensure that we capture entire questions (at risk of duplication)
    doc = SchemaDocument(page_content=content) if create_doc else content
    return RecursiveCharacterTextSplitter(chunk_size=LANGCHAIN_SPLITTER_CHUNK_SIZE,
                                          chunk_overlap=LANGCHAIN_SPLITTER_OVERLAP_SIZE).split_documents([doc])


def split_nltk(content) -> list:
    """
    Split content into sentences using NLTK.

    :param content: Content to split.
    :return: List of sentences.
    :rtype: list
    """

    return nltk.sent_tokenize(content)


def split_spacy(content) -> list:
    """
    Split content into sentences using spaCy.

    :param content: Content to split.
    :return: List of sentences.
    :rtype: list
    """

    doc = nlp(content)
    return [sent.text for sent in doc.sents]


def split(content, splitter: Callable = split_langchain, create_doc: bool = True) -> list:
    """
    Split content using a specified splitting function.

    :param content: Content to split.
    :param splitter: Function to use for splitting. Defaults to split_langchain.
    :type splitter: Callable
    :param create_doc: Flag to indicate whether to create a langchain Document for splitting.
    :type create_doc: bool
    :return: List of split content.
    :rtype: list
    """

    split_docs = splitter(content, create_doc)
    return [doc.page_content if not isinstance(doc, str) else doc for doc in split_docs]


def read_docx(file_path: str, splitter: Callable = split_langchain) -> list:
    """
    Parse a DOCX file into a list of content chunks using a specified splitting function.

    :param file_path: Path to the DOCX file.
    :type file_path: str
    :param splitter: Function to use for splitting the content. Defaults to split_langchain.
    :type splitter: Callable
    :return: List of split content chunks.
    :rtype: list
    """

    # use langchain/unstructured to parse the DOCX file
    loader = UnstructuredFileLoader(file_path, mode="elements")
    data = loader.load()

    # concatenate page contents, using double-linebreaks as page/element separators
    content = '\n\n'.join([page.page_content for page in data])
    content = clean_whitespace(content)

    return split(content, splitter)


def parse_csv(input_csv_file: str, splitter: Callable = split_langchain) -> dict | list:
    """
    Parse a CSV file into a dictionary with questionnaire data.

    The function handles REDCap data dictionaries explicitly, then falls back to generic handling of other CSV files.

    :param input_csv_file: Path to the CSV file.
    :type input_csv_file: str
    :param splitter: Function to use for splitting in case not a REDCap data dictionary. Defaults to split_langchain.
    :type splitter: Callable
    :return: Dictionary with form data or split content.
    :rtype: dict | list
    """

    form_data = {"questionnairedata": []}
    with open(input_csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        try:
            if 'Field Type' not in reader.fieldnames:
                # raise exception to fall back to generic CSV processing
                raise ValueError('CSV file does not appear to be a REDCap data dictionary (no "Field Type" column).')

            # process REDCap data dictionary
            for row in reader:
                question_data = {}
                field_type = row['Field Type']
                if field_type == 'descriptive' or field_type in ['text', 'number']:
                    question_data['question_id'] = row['Variable / Field Name']
                    question_data['question'] = row['Field Label']
                    question_data['instructions'] = row.get('Field Note', '')
                    question_data['options'] = []
                elif field_type in ['radio', 'checkbox']:
                    question_data['question_id'] = row['Variable / Field Name']
                    question_data['question'] = row['Field Label']
                    question_data['instructions'] = row.get('Field Note', '')
                    choices = row.get('Choices, Calculations, OR Slider Labels', '').split('|')
                    question_data['options'] = [{"value": c.split(',')[0].strip(), "label": c.split(',')[1].strip()} for
                                                c in choices]
                else:
                    continue
                form_data['question'].append(question_data)
        except Exception:
            # fall back to generic CSV processing
            content_list = ['\t'.join(row) for row in reader]
            content_str = '\n'.join(content_list)
            content_str = clean_whitespace(content_str)
            return split(content_str, splitter)

    return form_data


def convert_tables(md_text: str) -> str:
    """
    Convert Markdown tables to plain text format.

    :param md_text: Markdown text containing tables.
    :type md_text: str
    :return: Plain text representation of tables.
    :rtype: str
    """

    table_blocks = re.findall(r'\|[\s\S]+?\|', md_text)
    plain_text_tables = ""
    for table_block in table_blocks:
        rows = table_block.strip().split('\n')
        plain_text_tables += '\n'.join(
            ['\t'.join(cell.strip() for cell in row.split('|')[1:-1]) for row in rows]) + '\n\n'

    return plain_text_tables


def process_in_korn_format(data: list) -> dict:
    """
    Process data in a specific format to structure it into a dictionary with questions.

    :param data: List of modules and questions.
    :type data: list
    :return: Dictionary with structured questions.
    :rtype: dict
    """

    final_data = {'questionnairedata': []}
    for module in data:
        for question in module['questions']:
            final_data['questionnairedata'].append({
                'module': module.get('moduleTitle', ''),
                'question': question['question'],
                'instructions': question['instructions'],
                'options': question['options']
            })
    return final_data


def parse_xlsx(file_path: str, splitter: Callable = split_langchain):
    """
    Parse an XLSX file into a dictionary with questionnaire data.

    This function starts by assuming that the XLSX file is in XLSForm format, then falls back to generic parsing
    in the case where XLSForm parsing fails.

    :param file_path: Path to the XLSX file.
    :type file_path: str
    :param splitter: Function used to split content. Defaults to split_langchain.
    :type splitter: Callable
    :return: List of processed content in Document format or list format.
    """

    try:
        # define input tags for XML parsing
        input_tags = [
            "{http://www.w3.org/2002/xforms}input",
            "{http://www.w3.org/2002/xforms}select1",
            "{http://www.w3.org/2002/xforms}textarea",
            "{http://www.w3.org/2002/xforms}upload"
        ]

        # convert XLSX to XML (assuming XLSForm format)
        path_without_ext = os.path.splitext(file_path)[0]
        os.system(f'xls2xform "{file_path}" "{path_without_ext}.xml"')
        tree = ElementTree.parse(f"{path_without_ext}.xml")

        # initialize questionnaire processing
        questionnaire = []
        start = False
        state = ""

        # iterate over XML tree elements
        for elem in tree.iter():
            if elem.tag == "{http://www.w3.org/1999/xhtml}body":
                start = True
            if start:
                if elem.tag == "{http://www.w3.org/2002/xforms}group":
                    questionnaire.append({"module": "", "questions": []})
                    state = "group"
                if elem.tag in input_tags and not questionnaire:
                    questionnaire.append({"module": "", "questions": []})
                if elem.tag in ["{http://www.w3.org/2002/xforms}input", "{http://www.w3.org/2002/xforms}textarea",
                                "{http://www.w3.org/2002/xforms}upload"]:
                    state = "input"
                    questionnaire[-1]["questions"].append({"question": "", "instructions": "", "options": []})
                if elem.tag == "{http://www.w3.org/2002/xforms}select1":
                    state = "select"
                    questionnaire[-1]["questions"].append({"question": "", "instructions": "", "options": []})
                if elem.tag == "{http://www.w3.org/2002/xforms}item" and state == "select":
                    state = "option"
                    questionnaire[-1]["questions"][-1]["options"].append({"value": "", "label": ""})
                if elem.tag == "{http://www.w3.org/2002/xforms}label":
                    if state == "group":
                        questionnaire[-1]["module"] = elem.text
                    elif state in ["input", "select"]:
                        questionnaire[-1]["questions"][-1]["question"] = elem.text
                    elif state == "option":
                        questionnaire[-1]["questions"][-1]["options"][-1]["label"] = elem.text
                if elem.tag == "{http://www.w3.org/2002/xforms}value" and state == "option":
                    questionnaire[-1]["questions"][-1]["options"][-1]["value"] = elem.text
                    state = "select"
        return process_in_korn_format(questionnaire)
    except Exception:
        # fallback method for processing XLSX files (when XLSForm processing fails)
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        data = loader.load()
        content_list = []
        for page in data:
            html_content = page.metadata['text_as_html']
            page_name = page.metadata['page_name']
            content_list.append('<h1>' + page_name + '</h1>\n' + html_content)

        # split and return the processed content
        split_content = []
        for content in content_list:
            split_content.extend(split(clean_whitespace(content), splitter))
        return split_content


def read_local_html(path: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process local HTML file into a structured format.

    This function reads an HTML file from a local path, converts it to markdown,
    and then splits it into a structured format using a specified splitter function.

    :param path: Path to the local HTML file.
    :type path: str
    :param splitter: Function to split the processed markdown into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    """

    with open(path, 'r', encoding='utf-8') as file:
        doc = SchemaDocument(page_content=file.read())
        md = MarkdownifyHTMLProcessor().process(doc)
        return splitter(md, False)


def read_html(url: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process HTML content from a URL into a structured format.

    This function fetches HTML content from a given URL, converts it to markdown,
    and then splits it into a structured format using a specified splitter function.

    :param url: URL of the HTML page to be read.
    :type url: str
    :param splitter: Function to split the processed markdown into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    :rtype: list
    """

    response = requests.get(url)
    doc = SchemaDocument(page_content=response.text)
    md = MarkdownifyHTMLProcessor().process(doc)
    return splitter(md, False)


def read_pdf_pypdf(path: str, splitter: Callable = split_langchain, split_content: bool = True) -> list:
    """
    Read and process PDF content into a structured format using PyPDF.

    This function reads a PDF file, extracts text from each page, and optionally
    splits it into a structured format using a specified splitter function.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param split_content: Boolean flag to determine whether to split the content or not.
    :type split_content: bool
    :return: List of processed (and optionally split) content.
    :rtype: list
    """

    content = []
    reader = PdfReader(path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        content.append(text)
    return splitter('\n\n'.join(content)) if split_content else content


def extract_text_from_pdf_tabula(file_path: str, splitter: Callable = split_langchain,
                                 split_content: bool = True) -> list:
    """
    Extract and process text from a PDF file using Tabula for table extraction.

    This function reads a PDF file and extracts text from tables on each page using Tabula. It then cleans the
    extracted tables by dropping empty columns, filling NaN values, and combining them into a text format.
    Optionally, the extracted text can be split into a structured format using a specified splitter function.

    :param file_path: Path to the PDF file.
    :type file_path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param split_content: Boolean flag to determine whether to split the content or not.
    :type split_content: bool
    :return: List of processed (and optionally split) content from the PDF tables.
    :rtype: list
    """

    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)

    content = []
    for page in range(1, num_pages + 1):
        # read PDF tables with Tabula
        df = read_pdf(file_path, pages=page, multiple_tables=True)

        # clean and process tables
        cleaned_tables = clean_tables(df)

        # convert cleaned tables to text
        text = convert_tables_to_text(cleaned_tables)
        content.append(text)

    return splitter('\n\n'.join(content)) if split_content else content


def clean_tables(dataframes: list) -> list:
    """
    Clean and process extracted tables from a PDF.

    :param dataframes: List of DataFrame objects representing extracted tables.
    :type dataframes: list
    :return: List of cleaned and processed DataFrame objects.
    :rtype: list
    """

    cleaned_tables = []
    for table in dataframes:
        # drop columns with all NaN values and replace NaN with empty string
        table.dropna(axis=1, how='all', inplace=True)
        table.fillna('', inplace=True)
        # append non-empty tables to the list
        if not table.empty:
            cleaned_tables.append(table)
    return cleaned_tables


def convert_tables_to_text(tables: list) -> str:
    """
    Convert cleaned tables to a text format.

    :param tables: List of cleaned DataFrame objects representing tables.
    :type tables: list
    :return: String representation of the tables.
    :rtype: str
    """

    text = ''
    for table in tables:
        # convert DataFrame to string and replace multiple spaces
        table_string = table.to_string(index=False, header=True)
        table_string = re.sub(' +', ' ', table_string)
        text += table_string + '\n\n\n'
    return text


def read_ocr(path: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process text from an image-based PDF file using OCR.

    This function converts each page of a PDF to an image and then uses OCR to extract text.
    The extracted text is then split into a structured format using a specified splitter function.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    :rtype: list
    """

    pages = convert_from_path(path)
    content = ''
    for page in pages:
        text = pytesseract.image_to_string(page)
        content += text
    return splitter(content)


def read_pdf_combined(path: str, splitter: Callable = split_langchain, min_length: int = 600) -> list:
    """
    Read and process text from a PDF file combining PyPDF and Tabula extraction methods.

    This function extracts text from a PDF using both PyPDF and Tabula methods. The combined
    text is then checked for a minimum length. If the length requirement is met, the text is
    split into a structured format using a specified splitter function. If not, it falls back
    to OCR.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param min_length: Minimum expected length for the combined text.
    :type min_length: int
    :return: List of processed and split content, or content from OCR if length requirement is not met.
    :rtype: list
    """

    text_pypdf_per_page = read_pdf_pypdf(path, split_langchain, False)
    text_tabula_per_page = extract_text_from_pdf_tabula(path, split_langchain, False)

    combined_text = combine_texts(text_pypdf_per_page, text_tabula_per_page)

    # clean whitespace and check for minimum length
    combined_text = clean_whitespace(combined_text)
    if len(combined_text.strip()) >= min_length:
        return splitter(combined_text)

    return read_ocr(path, splitter)


def combine_texts(texts1: list, texts2: list) -> str:
    """
    Combine texts from two lists, appending text from the second list to the first.

    :param texts1: List of strings from the first source.
    :type texts1: list
    :param texts2: List of strings from the second source.
    :type texts2: list
    :return: Combined text as a single string.
    :rtype: str
    """

    combined_text = ""
    for text1, text2 in zip(texts1, texts2):
        combined_text += text1 + "\n" + text2 + "\n"
    return combined_text

"""
All File Reading functions go here
"""
import textract
import PyPDF2
from openAI import generate_keywords_from_notes
from PIL import Image
import pytesseract
import cv2


def ocr_with_tesseract(image_path: str) -> dict:
    image = Image.open(image_path)
    text: str = pytesseract.image_to_string(image)
    keywords: list = generate_keywords_from_notes(text)
    return {"filename": image_path, "content":text, "keywords": keywords}


def read_docx(docx_filename: str) -> dict:
    """
    :param docx_filename: The filename (and location) of the .docx document
    :return: Dictionary containing filename and ext content of the .docx file
    """
    text: str = textract.process(docx_filename).decode('utf-8')
    keywords = generate_keywords_from_notes(text)
    return {"filename": docx_filename, "content": text, "keywords": keywords}


def read_txt(txt_filename: str) -> dict:
    """

    :param txt_filename: The filename (and location) of the txt document
    :return: Dictionary containing filename and ext content of the .txt file
    """
    with open(txt_filename, 'r') as file:
        text = file.read()
        keywords = generate_keywords_from_notes(text)
        return {"filename": txt_filename, "content": text, "keywords": keywords}


def read_pdf(pdf_filename: str) -> dict:
    """

    :param pdf_filename: The filename (and location) of the pdf document
    :return: Dictionary containing filename and ext content of the .pdf file
    """
    with open(pdf_filename, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        content = ''
        for page in pdf.pages:
            content += page.extract_text()
    keywords = generate_keywords_from_notes(content)
    return {"filename": pdf_filename, "content": content, "keywords": keywords}

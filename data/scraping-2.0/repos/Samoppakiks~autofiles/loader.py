import os
import pdf2image
from PIL import Image
import pytesseract
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPDFLoader


def convert_pdf_to_images(pdf_path):
    return pdf2image.convert_from_path(pdf_path)


def save_images(pages):
    image_counter = 1
    for page in pages:
        filename = f"page_{image_counter}.jpg"
        page.save(filename, "JPEG")
        image_counter += 1
    return image_counter


def extract_text_from_images(image_counter):
    text = ""
    for i in range(1, image_counter):
        image_file = f"page_{i}.jpg"
        ocr_text = pytesseract.image_to_string(Image.open(image_file), lang="eng+hin")
        text += ocr_text + " "
        os.remove(image_file)
    return text


def extract_ocr(pdf_path):
    pages = convert_pdf_to_images(pdf_path)
    image_counter = save_images(pages)
    text = extract_text_from_images(image_counter)

    txt_file_path = (
        f"./extracted_texts/{os.path.splitext(os.path.basename(pdf_path))[0]}.txt"
    )
    with open(txt_file_path, "w") as file:
        file.write(text)

    return text


"""def extract_docx(docx_path):
    loader = Docx2txtLoader(docx_path)
    docs = loader.load()
    text = [doc.text for doc in docs]
    text_file_path = (
        f"./extracted_texts/{os.path.splitext(os.path.basename(docx_path))[0]}.txt"
    )
    with open(text_file_path, "w") as file:
        file.write(" ".join(text))
    return docs"""


docx_path = "/Users/saumya/Documents/Government/files/Producer Enterprise/Ujjala/Draft 6th PMC Minutes - Dairy Value Chain.docx"
text = extract_ocr(docx_path)
print(text)

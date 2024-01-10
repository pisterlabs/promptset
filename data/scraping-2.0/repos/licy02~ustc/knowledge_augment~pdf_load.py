# from langchain.document_loaders import PyPDFLoader
# pdf_path = "./中国氢能产业研究报告.pdf"
# loader = PyPDFLoader(pdf_path)
# pages = loader.load_and_split()
# text = pages[0].page_content

# from langchain.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader("./中国氢能产业研究报告.pdf")
# data = loader.load()
# print(data)

"""Loader that loads image files."""
# from typing import List
# from chinese_text_splitter import ChineseTextSplitter
# from langchain.document_loaders.unstructured import UnstructuredFileLoader
# from langchain.text_splitter import SpacyTextSplitter
# from paddleocr import PaddleOCR
# import os
# import fitz
# import nltk
# from configs.model_config import NLTK_DATA_PATH

# nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
#     """Loader that uses unstructured to load image files, such as PNGs and JPGs."""
#
#     def _get_elements(self) -> List:
#         def pdf_ocr_txt(filepath, dir_path="tmp_files"):
#             full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
#             if not os.path.exists(full_dir_path):
#                 os.makedirs(full_dir_path)
#             ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)
#             doc = fitz.open(filepath)
#             txt_file_path = os.path.join(full_dir_path, f"{os.path.split(filepath)[-1]}.txt")
#             img_name = os.path.join(full_dir_path, 'tmp.png')
#             with open(txt_file_path, 'w', encoding='utf-8') as fout:
#                 for i in range(doc.page_count):
#                     page = doc[i]
#                     text = page.get_text("")
#                     fout.write(text)
#                     fout.write("\n")
#
#                     img_list = page.get_images()
#                     for img in img_list:
#                         pix = fitz.Pixmap(doc, img[0])
#                         if pix.n - pix.alpha >= 4:
#                             pix = fitz.Pixmap(fitz.csRGB, pix)
#                         pix.save(img_name)
#
#                         result = ocr.ocr(img_name)
#                         ocr_result = [i[1][0] for line in result for i in line]
#                         fout.write("\n".join(ocr_result))
#             if os.path.exists(img_name):
#                 os.remove(img_name)
#             return txt_file_path
#
#         txt_file_path = pdf_ocr_txt(self.file_path)
#         from unstructured.partition.text import partition_text
#         return partition_text(filename=txt_file_path, **self.unstructured_kwargs)

def extract_text_from_pdf(file_path: str):
    """Extract text content from a PDF file."""
    import PyPDF2
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ''
            for text in raw_text:
                new_text += text
                if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';',  '”', '’', '）', '】', '》', '」',
                                '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                    contents.append(new_text)
                    new_text = ''
            if new_text:
                contents.append(new_text)
    return contents

# def extract_text_from_txt(file_path: str):
#     """Extract text content from a TXT file."""
#     contents = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         contents = [text.strip() for text in f.readlines() if text.strip()]
#     return contents
#
# def extract_text_from_docx(file_path: str):
#     """Extract text content from a DOCX file."""
#     import docx
#     document = docx.Document(file_path)
#     contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
#     return contents
#
# def extract_text_from_markdown(file_path: str):
#     """Extract text content from a Markdown file."""
#     import markdown
#     from bs4 import BeautifulSoup
#     with open(file_path, 'r', encoding='utf-8') as f:
#         markdown_text = f.read()
#     html = markdown.markdown(markdown_text)
#     soup = BeautifulSoup(html, 'html.parser')
#     contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
#     return contents
#
# def LoadPdf(file):
#     loader = UnstructuredPaddlePDFLoader(file, mode="elements")
#     docs = loader.load_and_split()
#     return docs

# contents = extract_text_from_pdf('中科大不完全入学指南.pdf')
# print(contents[-1])
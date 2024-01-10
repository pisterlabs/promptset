#!/usr/bin/python3
""" TEXT EXTRACTOR FROM PDF, PNG, JPG """
# import cv2
# import easyocr
import PyPDF2

from .openai_request import openai_call


def pdf_to_text(filename: str):
    """Extracts text from pdf files

    Args:
        filename: directory containing file

    Return:
        object: json
    """
    pdfFileObj = open(filename, 'rb')

    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    output = ""
    response = ""
    keyword = """Match only transaction details with a date attached in this text below into a table. No title.
                Response only as JSON format inside an array.\n Example: [{"key_response":"value_response"}]"""
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        output = pageObj.extractText()
        output = output.strip('\n')
        output = output[:2000]
        prompt = f"{keyword}\n\n{output}\n\n"
        result = openai_call(prompt, 0.1)
        response += result
    pdfFileObj.close()
    return response.strip()


class OCR_Reader:
    """Initialize the reader with a jpg image.

    Parameters:
        - image/frame: numpy array
        - languages: list of languages to use for OCR, default is ['en', 'it']
    """

    def __init__(self, gpu=False, languages=["en", "it"]):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def read_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adapted = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            85,
            11)
        result = self.reader.readtext(gray)
        text = []
        boxes = []
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text.append(detection[1])
            boxes.append(f"Box: {top_left + bottom_right}")
            try:
                image = cv2.rectangle(
                    image, top_left, bottom_right, (0, 255, 0), 2)
            except BaseException:
                continue
        return image, text, boxes

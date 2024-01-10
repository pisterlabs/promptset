from pdf2image import convert_from_path
from pytesseract import *
from PIL import Image

# import pyPdf as pdf
import os
import openai

import cv2

def convert_pdf_to_img(pdf_file):
    return convert_from_path(pdf_file)

def convert_image_to_text(file):
    # config = "-l eng+jpn+kor+rus+chi_sim+vie+thai"
    text = pytesseract.image_to_string(file, lang="eng")
    # text = pytesseract.image_to_string(file, config=config)
    return text

def get_text_from_any_pdf(pdf_file):

    images = convert_pdf_to_img(pdf_file)
    final_text = ""
    for pg, img in enumerate(images):
        
        final_text += convert_image_to_text(img)

    return final_text

path_to_pdf = './PDF/img_sample13.png'
print(get_text_from_any_pdf(path_to_pdf)) # pdf text 추출
#print(convert_image_to_text(path_to_pdf))


#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


import PyPDF2
import os



from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


import openai
from transformers import pipeline

# Import the necessary dependencies: PyPDF2 for PDF processing and OpenAI for interfacing with GPT-3.5-turbo.

pdf_summary_text = ""

# Initialize an empty string to store the summarized text.

pdf_file_path = "./temp_dir/"

rsrcmgr = PDFResourceManager()
retstr = StringIO()
# codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, retstr, laparams=laparams)

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

# Open the PDF file and create a PyPDF2 reader object.

for filename in os.listdir(pdf_file_path):
    input(filename)
    file_path = os.path.join(pdf_file_path, filename)
    if os.path.isfile(file_path):
        fp = open(file_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        pdf_text = retstr.getvalue()
        fp.close()

        for i in range(0, len(pdf_text)):
            text = pdf_text[i : i+10000]
            summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
            print(summary)
            i = i +10000


device.close()
retstr.close()